import numpy as np
import pandapower as pp
import pandapower.networks as pn
import timeit
import copy


rede = pn.case14()


def gerenciar_rede(rede):
    
    """
    Esta funcao organiza a rede obtida do PandaPower de modo que ela possa ser mais facilmente utilizada pelo algoritmo.
    Suas funcionalidades são:
        
        -> Ordenar os parâmetros da rede (v_bus, tap, shunt, etc) por índices;
        -> Obter os transformadores com controle de tap;
        -> Gerar o vetor com os valores dos taps dos transformadores;
        -> Gerar os vetores com os valores discretos para os shunts de cada sistema;
        -> Gerar o primeiro agente de busca (que contém as variáveis do ponto de operação do sistema);
        -> Obter as condutâncias das linhas.
        
    Input:
        -> rede
        
    Output:
        
        -> rede gerenciada (não devolvida, salva diretamente na variável rede);
        -> primeiro agente de buscas: lobo_1;
        -> vetor de condutâncias da rede: G_rede;
        -> matriz das linhas de transmissao: linhas
    """
    
    #Ordenar os índices da rede
    rede.bus = rede.bus.sort_index()
    rede.res_bus = rede.res_bus.sort_index()
    rede.gen = rede.gen.sort_index()
    rede.line = rede.line.sort_index()
    rede.shunt = rede.shunt.sort_index()
    rede.trafo = rede.trafo.sort_index()
    
    
    #Dicionário que contem os valores dos shunts para cada sistema IEEE
    valores_shunts = {"14": np.array([0, 0.19, 0.34, 0.39]),
                      "30": np.array([[0, 0.19, 0.34, 0.39],
                                      [0, 0.05, 0.09]]),
                      "57": np.array([[0, 0.12, 0.22, 0.27], 
                                      [0, 0.04, 0.07, 0.09], 
                                      [0, 0.1, 0.165]]),
                      "118": np.array([[-0.4, 0],
                                       [0, 0.06, 0.07, 0.13, 0.14, 0.2],
                                       [-0.25, 0],
                                       [0, 0.1],
                                       [0, 0.1],
                                       [0, 0.1],
                                       [0, 0.15],
                                       [0, 0.08, 0.12, 0.2],
                                       [0, 0.1, 0.2],
                                       [0, 0.1, 0.2],
                                       [0, 0.1, 0.2],
                                       [0, 0.1, 0.2],
                                       [0, 0.06, 0.07, 0.13, 0.14, 0.2],
                                       [0, 0.06, 0.07, 0.13, 0.14, 0.2]]),
                      "300": np.array([[0, 2, 3.5, 4.5],
                                       [0, 0.25, 0.44, 0.59],
                                       [0, 0.19, 0.34, 0.39],
                                       [-4.5, 0],
                                       [-4.5, 0],
                                       [0, 0.25, 0.44, 0.59],
                                       [0, 0.25, 0.44, 0.59],
                                       [-2.5, 0],
                                       [-4.5, 0],
                                       [-4.5, 0],
                                       [-1.5, 0],
                                       [0, 0.25, 0.44, 0.59],
                                       [0, 0,15],
                                       [0, 0.15]])
                      }
    
    #Vetor que contém os valores discretos que os taps podem assumir
    #Observar se o shape (11,) não trará problemas
    valores_taps = np.arange(start = 0.9, stop = 1.1, step = 0.00625)
    
    """
    Matriz contendo as linhas de transmissão da rede:
        -> linhas[0] = vetor com as barras de ínicio;
        -> linhas[1] = vetor com as barras de términa;
        -> linhas[2] = vetor com as resistências em pu (r_pu) das linhas;
        -> linhas[3] = vetor com as reatâncias em pu (x_pu) das linhas.
    r_pu = r_ohm/z_base
    x_pu = x_ohm/z_base
    g = r_pu/(r_pu^2 + x_pu^2)
    """
    linhas = np.zeros((4, rede.line.index[-1]+1))
    linhas[0] = rede.line.from_bus.to_numpy()
    linhas[1] = rede.line.to_bus.to_numpy()
    v_temp = rede.bus.vn_kv.to_numpy()
    z_base = np.power(np.multiply(v_temp,1000), 2)/100e6
    for i in range(rede.line.index[-1]+1):
        linhas[2][i] = rede.line.r_ohm_per_km[i]/z_base[int(linhas[0][i])]
        linhas[3][i] = rede.line.x_ohm_per_km[i]/z_base[int(linhas[0][i])]
    del v_temp, z_base
    
    #Vetor G_rede com as condutâncias das linhas de transmissão
    G_rede = np.zeros((1, rede.line.index[-1]+1))
    G_rede = np.divide(linhas[2], np.power(linhas[2],2)+np.power(linhas[3],2))
    
    
    
    """
    O primeiro lobo (agente de busca) será inicializado com os valores de operação da rede fornecidos
    pelo PandaPower: vetor lobo_1.
    
    tap_pu = (tap_pos + tap_neutral)*tap_step_percent/100 (equação fornecida pelo PandaPower)
    shunt_pu = -100*shunt (equação fornecida pelo PandaPower)
    
    As variáveis v_temp, taps_temp e shunt_temp são utilizadas para receber os valores de tensão, tap e shunt da rede
    e armazenar no vetor lobo_1
    """
    
    #num_trafo_controlado: variavel para armazenar o número de trafos com controle de tap
    num_trafo_controlado = rede.trafo.tap_pos.count()
    
    #num_barras : variavel utilizada para salvar o numero de barras do sistema
    num_barras = rede.bus.name.count()
    
    #num_shunt: variavel para armazenar o numero de shunts do sistema
    num_shunt = rede.shunt.in_service.count()
    
    v_temp = rede.gen.vm_pu.to_numpy(dtype = 'float64')
    
    taps_temp = 1 + ((rede.trafo.tap_pos.to_numpy()[0:num_trafo_controlado] +\
                      rede.trafo.tap_neutral.to_numpy()[0:num_trafo_controlado]) *\
                     (rede.trafo.tap_step_percent.to_numpy()[0:num_trafo_controlado]/100))
        
    shunt_temp = -rede.shunt.q_mvar.to_numpy()/100
    
    lobo_1 = np.array([np.concatenate((v_temp, taps_temp, shunt_temp),axis=0)])
    del v_temp, taps_temp, shunt_temp
    
    matriz_G = np.zeros((num_barras,num_barras))
    matriz_G[linhas[0].astype(np.int), linhas[1].astype(np.int)] = G_rede 
    
    parametros_rede = {"Linhas": linhas,
                       "G": G_rede,
                       "Lobo1": lobo_1,
                       "Valores_shunts": valores_shunts,
                       "Valores_taps": valores_taps,
                       "num_barras": num_barras,
                       "num_trafo_controlado": num_trafo_controlado,
                       "num_shunt": num_shunt,
                       "matriz_G": matriz_G}
    

    return parametros_rede
    


def fluxo_de_carga(rede, agente):
    pass


def funcao_objetivo(rede, matriz_G):
    """
    O objetivo desta função é calcular a função objetivo para o problema de FPOR.
    A função objetivo deste problema dá as perdas de potência ativa no SEP.
    
    f = sum g_km * [v_k^2 + v_m^2 - 2*v_k*v_m*cos(theta_k - theta_m)]
    
    """
    v_k = np.array([rede.res_bus.vm_pu.to_numpy(dtype = np.float64)])
    v_m = v_k.T
    
    theta_k = np.radians(np.array([rede.res_bus.va_degree.to_numpy(dtype = np.float64)]))
    theta_m = theta_k.T
    
    f = np.power(v_k,2) + np.power(v_m,2) - 2*np.multiply(np.multiply(v_k,v_m),np.cos(theta_k-theta_m))
    
    f = np.multiply(matriz_G, f)
    
    f=np.squeeze(np.sum(np.array([f[np.nonzero(f)]])))
    
    return f


def penalidade_v(rede, agente):
    pass

def penalidade_senoidal_tap(agente):
    pass
    
def penalidade_senoidal_shunt(agente):
    pass

def inicializar_alcateia(n_lobos, dim, t_max):
    pass

def otimizar_alcateia(f_obj, pen_v, pen_tap, pen_shunt, lbd):
    pass

def visualizar_resultados():
    pass


"""
----------------------------------------------------------TESTES ---------------------------------------------------------
"""
def a():
    
    # def funcao_objetivo2(rede, params):
    #     """
    #     O objetivo desta função é calcular a função objetivo para o problema de FPOR.
    #     A função objetivo deste problema dá as perdas de potência ativa no SEP.
        
    #     f = sum g_km * [v_k^2 + v_m^2 - 2*v_k*v_m*cos(theta_k - theta_m)]
        
    #     """
    #     v_k = np.array([rede.res_bus.vm_pu[params["Linhas"][0].astype(np.int)].to_numpy(dtype=np.float64)])
    #     v_m = np.array([rede.res_bus.vm_pu[params["Linhas"][1].astype(np.int)].to_numpy(dtype=np.float64)])
    #     theta_k = np.radians(np.array([rede.res_bus.va_degree[params["Linhas"][0].astype(np.int)].to_numpy(dtype=np.float64)]))
    #     theta_m = np.radians(np.array([rede.res_bus.va_degree[params["Linhas"][1].astype(np.int)].to_numpy(dtype=np.float64)]))
    #     g = np.array([params["G"]])
    #     #temp1 = np.power(v_k,2) + np.power(v_m,2) - 2*np.multiply(np.multiply(v_k,v_m),np.cos(theta_k-theta_m))
    #     #temp2 = np.multiply(matriz_G, temp1)
    #     #f=np.squeeze(np.sum(np.array([temp2[np.nonzero(temp2)]])))
    #     f = np.square(v_k) + np.square(v_m) - 2*np.multiply(np.multiply(v_k,v_m),np.cos(theta_k-theta_m))
    #     f = np.squeeze(np.sum(np.multiply(g,f)))
    #     return f
    
    # redes = {"1": copy.copy(rede),
    #          "2": copy.copy(rede),
    #          "3": copy.copy(rede),
    #          "4": copy.copy(rede),
    #          "5": copy.copy(rede)}
    
    # def fluxo_sequencial(redes):
    #     for key in redes:
    #         pp.runpp(redes[key], algorithm='fdbx')
    
    # def fluxo_paralelo_multi(redes):
    #     pp.runpp(rede, algorithm = 'fdbx')
    
    # def fluxo_paralelo_ray(rede):
    #     pp.runpp(rede, algorithm='fdbx')
    
    
    # t1 = time.time()
    # fluxo_sequencial(redes)
    # t2= time.time()
    # print("Fluxo sequencial: " + str(t2-t1) + " s")
    
    # t3 = time.time()
    # if __name__ == '__main__':
    #     with multiprocessing.Pool(processes=num) as p:
    #         p.map(fluxo_paralelo_multi, rop)
    #         p.close()
    # t4 = time.time()
    # print("Fluxo paralelo: " + str(t4 - t3) + " s")
    
    
    # rop = [(redes["1"], p1["matriz_G"]),
    #         (redes["2"], p1["matriz_G"]),
    #         (redes["3"], p1["matriz_G"]),
    #         (redes["4"], p1["matriz_G"]),
    #         (redes["5"], p1["matriz_G"])]
    
    # pp.runpp(rede, algorithm='fdbx')
    
    
    # a = funcao_objetivo2(rede, r1)
    # b = funcao_objetivo(rede, r1["matriz_G"])
    # print(a==b)
    
    # t_1 = []
    # for i in range(1000):
    #     t11 = timeit.default_timer()
    #     funcao_objetivo(rede, r1["matriz_G"])
    #     t12 = timeit.default_timer()
    #     t1= t12-t11
    #     t_1.append(t1)
    # vec = min(t_1)
    # print("F_obj 1: " + str(vec) + " s")
    
    # t_2 = []
    # for j in range(1000):
    #     t21 = timeit.default_timer()
    #     funcao_objetivo2(rede, r1)
    #     t22 = timeit.default_timer()
    #     t2= t22-t21
    #     t_2.append(t2)
    # vec2 = min(t_2)
    # print("F_obj 2: " + str(vec2) + " s")
    return 0

r1 = gerenciar_rede(rede)
pp.runpp(rede, algorithm='fdbx')




