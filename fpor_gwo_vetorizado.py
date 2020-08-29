import numpy as np
import pandapower as pp
import pandapower.networks as pn
import timeit
import copy


'''--------------------------------------------- Funções auxiliares ---------------------------------------------------'''
'''
Ver 

-> https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array/2566508
-> https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
'''

def teste():
    lst = [1.5, 2.5, 3.5, 4.5]
    valores = [1, 2, 3, 4, 5]
    for valor in valores:
        print('Sup: {}'.format(discreto_superior(valor, lst)))
        print('Inf: {}'.format(discreto_inferior(valor, lst)))

#Até agora, parece que funciona
def discreto_superior(valor, lista):
    '''
    Função que retorna o valor discreto superior mais próximo de x em uma lista
    
    Inputs:
        -> valor = número real
        -> lista = numpy.array
    
    Ouputs:
        -> valor discreto de superior de 'lista' mais próximo de 'valor'
    '''
    #Garante que a lista seja uma array numpy
    lista = np.asarray(lista, dtype = np.float32)
    
    if valor >= lista[-1]:
        return lista[-1]
    else:
        return np.squeeze(lista[np.searchsorted(a=lista, v=np.array([valor]), side = 'right')])

def discreto_inferior(valor, lista):
    '''
    Função que retorna o valor discreto inferior mais próximo de x em uma lista
    
    Inputs:
        -> valor = número real
        -> lista = numpy.array
    
    Ouputs:
        -> valor discreto inferior de 'lista' mais próximo de 'valor'
    '''
    lista = np.asarray(lista, dtype = np.float32)
    
    if valor <= lista[0]:
        return lista[0]
    else:
        return np.squeeze(lista[np.searchsorted(a=lista, v=np.array([valor]), side = 'left') - 1])
    

'''--------------------------------------------- Funções principais ---------------------------------------------------'''

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
    
    
    
    #num_trafo_controlado: variavel para armazenar o número de trafos com controle de tap
    num_trafo_controlado = rede.trafo.tap_pos.count()
    
    #num_barras : variavel utilizada para salvar o numero de barras do sistema
    num_barras = rede.bus.name.count()
    
    #num_shunt: variavel para armazenar o numero de shunts do sistema
    num_shunt = rede.shunt.in_service.count()
    
    #num_gen: variavel para armazenar o número de barras geradoras do sistema
    num_gen = rede.gen.in_service.count()
    
    '''
    Cria as varíaveis globais nb, nt, ns, ng para facilitar o uso desses parâmetros em outros funções
    
    *Potencialmente desastroso*
    '''
    global nb, nt, ns, ng
    nb, nt, ns, ng = num_barras, num_trafo_controlado, num_shunt, num_gen
    
    '''
    Muda os valores máximos e mínimos permitidos das tensões das barras dos sistemas de 118 e 300 barras:
        min_vm_pu: 0.94 -> 0.90
        max_vm_pu: 1.06 -> 1.10
    '''
    if num_barras == 118 or num_barras == 300:
        rede.bus.min_vm_pu = 0.90
        rede.bus.max_vm_pu = 1.10
    
    #Dicionário que contem os valores dos shunts para cada sistema IEEE
    valores_shunts = {"14": [[0, 0.19, 0.34, 0.39]],
                      "30": [[0, 0.19, 0.34, 0.39],
                             [0, 0.05, 0.09]],
                      "57": [[0, 0.12, 0.22, 0.27], 
                             [0, 0.04, 0.07, 0.09], 
                             [0, 0.1, 0.165]],
                      "118": [[-0.4, 0],
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
                              [0, 0.06, 0.07, 0.13, 0.14, 0.2]],
                      "300": [[0, 2, 3.5, 4.5],
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
                              [0, 0.15]]
                      }
    
    #Vetor que contém os valores discretos dos taps: entre 0.9 e 1.1 com passo = tap_step
    #Precisa ser um tensor de rank 1 para que a alcateia possa ser inicializada
    global tap_step
    tap_step = 0.00625
    valores_taps = np.arange(start = 0.9, stop = 1.1, step = tap_step)
    
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
    G_rede = np.array([np.divide(linhas[2], np.power(linhas[2],2)+np.power(linhas[3],2))])
    
    #Matriz de condutância nodal da rede. É equivalente à parte real da matriz de admintância nodal do sistema
    matriz_G = np.zeros((num_barras,num_barras))
    matriz_G[linhas[0].astype(np.int), linhas[1].astype(np.int)] = G_rede 
    
    """
    O primeiro lobo (agente de busca) será inicializado com os valores de operação da rede fornecidos
    pelo PandaPower: vetor lobo_1.
    
    tap_pu = (tap_pos + tap_neutral)*tap_step_percent/100 (equação fornecida pelo PandaPower)
    shunt_pu = -100*shunt (equação fornecida pelo PandaPower)
    
    As variáveis v_temp, taps_temp e shunt_temp são utilizadas para receber os valores de tensão, tap e shunt da rede
    e armazenar no vetor lobo_1
    """
    
    v_temp = rede.gen.vm_pu.to_numpy(dtype = 'float64')
    
    taps_temp = 1 + ((rede.trafo.tap_pos.to_numpy()[0:num_trafo_controlado] +\
                      rede.trafo.tap_neutral.to_numpy()[0:num_trafo_controlado]) *\
                     (rede.trafo.tap_step_percent.to_numpy()[0:num_trafo_controlado]/100))
        
    shunt_temp = -rede.shunt.q_mvar.to_numpy()/100
    
    lobo_1 = np.array([np.concatenate((v_temp, taps_temp, shunt_temp),axis=0)])
    del v_temp, taps_temp, shunt_temp
      
    parametros_rede = {"Linhas": linhas,
                       "G": G_rede,
                       "Lobo1": lobo_1,
                       "Valores_shunts": valores_shunts,
                       "Valores_taps": valores_taps,
                       "num_barras": num_barras,
                       "num_trafo_controlado": num_trafo_controlado,
                       "num_shunt": num_shunt,
                       "num_gen": num_gen,
                       "matriz_G": matriz_G}

    return parametros_rede
    


def fluxo_de_carga(rede, agente):
    pass

def funcao_objetivo_e_pen_v(rede, matriz_G, v_lim_sup, v_lim_inf):
    """
    E função  calcula a função objetivo para o problema de FPOR e também calcula
    a penalização das tensões que ultrapassam o limite superior ou ficam abaixo do limite
    inferior.
    
    A função objetivo deste problema dá as perdas de potência ativa no SEP.
    
    f = sum g_km * [v_k^2 + v_m^2 - 2*v_k*v_m*cos(theta_k - theta_m)]
    
    A penalização das tensões é:
        
    pen_v = sum(v - v_lim)^2
        v_lim = v_lim_sup, se v > v_lim_sup
        v_lim = v, se v_lim_inf < v < v_lim_sup
        v_lim = v_lim_inf, se v < v_lim_inf
    
    """
    #Talvez tenha uma melhor implementação de pen_v
    
    
    v_k = np.array([rede.res_bus.vm_pu.to_numpy(dtype = np.float64)])
    v_m = v_k.T
    
    theta_k = np.radians(np.array([rede.res_bus.va_degree.to_numpy(dtype = np.float64)]))
    theta_m = theta_k.T
    
    f = np.power(v_k,2) + np.power(v_m,2) - 2*np.multiply(np.multiply(v_k,v_m),np.cos(theta_k-theta_m))
    
    f = np.multiply(matriz_G, f)
    
    f=np.squeeze(np.sum(np.array([f[np.nonzero(f)]])))
    
    #Tensões que ultrapassaram o limite superior
    v_up = np.multiply(v_k, np.greater(v_k, v_lim_sup))
    
    #Limites ultrapassados
    v_lim_up = np.multiply(v_lim_sup, np.greater(v_k, v_lim_sup))
    
    #Tensões que estão abaixo do limite inferior
    v_down = np.multiply(v_k, np.less(v_k, v_lim_inf))
    
    #Limites violados
    v_lim_down = np.multiply(v_lim_inf, np.greater(v_k, v_lim_inf))
    
    pen_v = np.squeeze(np.sum(np.square(v_up - v_lim_up) + np.square(v_down - v_lim_down)))
    
    return f, pen_v


#Testada em um notebook
def penalidade_senoidal_tap(alcateia):
    '''
    Esta função retorna a penalidade senoidal sobre os taps dos transformadores para toda a alcateia.
    Dado um tap t e um passo discreto s, a função penalidade senoidal é dada por:
        pen_sen_tap = sum {sen^2(t*pi/s)}    
    
    Inputs:
        -> alcateia
        
    Outputs:
        -> pen_taps: um vetor cuja forma é (1, n_lobos) contendo as penalizações referentes aos taps para toda a alcateia
    '''
    
    #Variável para receber os taps de todos os lobos
    taps = alcateia[ng:ng+nt, :]
    
    #Variável para armazenar as penalidades referentes aos taps de todos os lobos
    pen_taps = np.array(np.zeros((1, alcateia.shape[1])))
    
    #Executa a equação da penalização sem efetuar a soma
    taps = np.power(np.sin(taps*np.pi/tap_step) , 2)
    
    #Executa a soma ao longo das colunas da variável taps
    pen_taps = np.sum(taps, axis=0)
    
    #Deleta a variável taps
    del taps
    
    return pen_taps


def penalidade_senoidal_shunt(alcateia):
    pass

def inicializar_alcateia(n_lobos, parametros_rede):
    '''
    Esta função inicializa a alcateia de lobos (agentes de busca) como uma matriz (numpy array) com formato (dim+6, n_lobos)
    
        -> Cada coluna da matriz representa um lobo.
        -> As linhas de 0 a dim são as variáveis do problema.
        -> A linha dim+1 armazena a função fitness para cada lobo;
        -> A linha dim+2 armazena a função objetivo para cada lobo;
        -> A linha dim+3 armazena um caracter que indica a posição hierárquica do lobo:
            + 1 = lobo alfa;
            + 2 = lobo beta;
            + 3 = lobo delta;
            + 4 = lobo omega.
        -> A linha dim+4 armazena a penalidade dos taps para cada lobo;
        -> A linha dim+5 armazena a penalidade dos shunts para cada lobo;
        -> A linha dim+6 armazena a penalidade das tensões para cada lobo;
        
    Inputs:
        -> n_lobos = número de agentes de busca;
        -> parametros_rede - obtido via função gerenciar_rede
    
    Output:
        -> Alcateia
    '''
    # nb = parametros_rede["num_barras"]
    # nt = parametros_rede["num_trafo_controlado"]
    # ns = parametros_rede["num_shunt"]
    # ng = parametros_rede["num_gen"]
    
    #Variável que armazena o número de variáveis do problema
    dim = ng+nt+ns
    
    #Inicialização da Alcateia com zeros
    alcateia = np.zeros((dim+6, n_lobos))
    
    #Inicialização aleatória das variáveis contínuas (tensões das barras geradoras) a partir de uma distribuição normal
    alcateia[:ng, :] = np.random.uniform(rede.bus.min_vm_pu[0], rede.bus.max_vm_pu[1], size=(ng,n_lobos))
    
    #Inicialização dos taps dos transformadores a partir da escolha aleatória dentro dos valores discretos permitidos
    alcateia[ng:ng+nt, :] = np.random.choice(parametros_rede["Valores_taps"], size =(nt, n_lobos))
    
    #Inicialização dos shunts das barras a partir da escolha aleatória dentro dos valores discretos permitidos
    #Não consegui escapar do loop de for aqui ainda =/
    for i in range(ns):
        alcateia[ng+nt+i, :] = np.random.choice(parametros_rede["Valores_shunts"][str(nb)][i], size = (ns,n_lobos))
    
    #Inicializar a função fitness e a função objetivo de cada lobo
    alcateia[dim:dim+2, :] = np.inf
    
    #Inicializar a posição de cada lobo como delta
    alcateia[dim+2, :] = 4
    
    #Inicializar as penalidades de cada lobo
    alcateia[dim+3:, :] = 0
    
    #Inserir o lobo_1
    alcateia[:dim, 0] = parametros_rede["Lobo1"]
    
    return alcateia


def otimizar_alcateia(f_obj, pen_v, pen_tap, pen_shunt, lbd, t_max):
    pass

def visualizar_resultados():
    pass


"""
---------------------------------------------------------- TESTES ---------------------------------------------------------
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

# rede = pn.case14()
# r1 = gerenciar_rede(rede)
# pp.runpp(rede, algorithm='fdbx')
# alcateia = inicializar_alcateia(12, r1)




