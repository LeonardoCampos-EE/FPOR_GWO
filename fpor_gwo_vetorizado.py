import numpy as np
import timeit
import copy
import pandapower as pp
from pandapower.networks import case14
import matplotlib.pyplot as plt

rede = case14()

#Executar o fluxo de carga uma primeira vez acelera os cálculos posteriores
pp.runpp(rede, algorithm = 'nr', numba = True)
'''--------------------------------------------- Funções auxiliares ---------------------------------------------------'''


def discreto_superior(vetor_x, lista_discretos):
    '''
    Função que retorna o valor discreto superior de 'lista_discretos' mais próximo de todos os valores x de 'vetor_x' 

    Inputs:
        -> vetor_x = vetor (numpy array) contendo os valores que deseja-se obter o número discreto mais próximos
        -> lista_discretos = lista (python list) que contém o conjunto de valores discretos que cada variável x admite
    
    Ouputs:
        -> x_sup = vetor (numpy array) contendo os valores discretos superiores de 'lista_discretos' mais próximo 
           dos valores de 'vetor_x'
    '''
    #Vetor de saída da função. Possui o mesmo formato (shape) que vetor_x
    x_sup = np.zeros(vetor_x.shape)
    
    #Cópia de 'vetor_x'. Esta cópia é feita para evitar erros de alocamento dinâmico de memória.
    vetor = np.copy(vetor_x)
    
    #GAMBIARRA SEM A QUAL O CÓDIGO NÃO FUNCIONA PROS VALORES DE 'vetor_x' que pertençam a 'lista_discretos'
    #vetor = vetor + 1e-3
    
    #Garante que a lista seja uma array numpy e armazena o resultado na variável 'lista'
    lista = np.asarray(lista_discretos, dtype = np.float32)
    
    '''
    Garante que os valores de 'vetor_x' estejam dentro dos limites de 'lista discretos' por um pequeno fator de 10^-3.
    Caso contrário, a função numpy.searchsorted descrita a frente resultará em erro.
    '''
    np.clip(a = vetor, a_min = lista[0]+1e-3, a_max = lista[-1]-1e-3, out = vetor)
    
    '''
    Utilizando a função numpy.searchsorted() para buscar os índices de 'lista_discretos' que correspondem aos valores
    discretos superiores aos valores de 'vetor_x'
    '''
    indices = np.searchsorted(a=lista, v = vetor, side='right')
    
    #Armazena os valores de 'lista_discretos' cujos índices correspondem aos discretos superiores de 'vetor_x'
    x_sup = np.take(lista, indices)
    
    #Deleta as variáveis locais
    del vetor, lista, indices
    
    return x_sup

def discreto_inferior(vetor_x, lista_discretos):
    '''
    Função que retorna o valor discreto inferior de 'lista_discretos' mais próximo de todos os valores x de 'vetor_x' 
    
    Inputs:
        -> vetor_x = vetor (numpy array) contendo os valores que deseja-se obter o número discreto mais próximo
        -> lista_discretos = lista (python list) que contém o conjunto de valores discretos que cada variável x admite
    
    Ouputs:
        -> x_inf = vetor (numpy array) contendo os valores discretos inferiores de 'lista_discretos' mais próximos 
           dos valores de 'vetor_x'
    '''
    
    #Vetor de saída da função. Possui o mesmo formato (shape) que vetor_x
    x_inf = np.zeros(vetor_x.shape)
    
    #Cópia de 'vetor_x'. Esta cópia é feita para evitar erros de alocamento dinâmico de memória.
    vetor = np.copy(vetor_x)
    
    #GAMBIARRA SEM A QUAL O CÓDIGO NÃO FUNCIONA PROS VALORES DE 'vetor_x' que pertençam a 'lista_discretos'
    #vetor = vetor - 1e-3
    
    #Garante que a lista seja uma array numpy e salva o resultado na variável local 'lista'
    lista = np.asarray(lista_discretos, dtype = np.float32)
    
    '''
    Garante que os valores de 'vetor_x' estejam dentro dos limites de 'lista discretos' por um pequeno fator de 10^-3.
    Caso contrário, a função numpy.searchsorted descrita a frente resultará em erro. Salva o resultado de numpy.clip
    na variável local 'vetor'
    '''
    np.clip(a = vetor, a_min = lista_discretos[0]+1e-3, a_max = lista_discretos[-1]-1e-3, out = vetor)
    '''[np.nonzero(vetor)]'''
    
    
    '''
    Utilizando a função numpy.searchsorted() para buscar os índices de 'lista_discretos' que correspondem aos valores
    discretos inferiores aos valores de 'vetor_x'
    '''
    indices = np.searchsorted(a=lista, v = vetor, side='left') - 1
    
    #Armazena os valores de 'lista_discretos' cujos índices correspondem aos discretos superiores de 'vetor_x'
    x_inf = np.take(lista, indices)
    
    #Deleta as variáveis locais
    del vetor, lista, indices
    
    return x_inf


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
    

def funcao_objetivo_e_pen_v(rede, matriz_G, v_lim_sup, v_lim_inf):
    """
    E função  calcula a função objetivo para o problema de FPOR e também calcula
    a penalização das tensões que ultrapassam o limite superior ou ficam abaixo do limite
    inferior para cada agente de busca.
    
    Esta função não é vetorizada para toda a alcateia
    
    A função objetivo deste problema dá as perdas de potência ativa no SEP.
    
    f = sum g_km * [v_k^2 + v_m^2 - 2*v_k*v_m*cos(theta_k - theta_m)]
    
    A penalização das tensões é:
        
    pen_v = sum(v - v_lim)^2
        v_lim = v_lim_sup, se v > v_lim_sup
        v_lim = v, se v_lim_inf < v < v_lim_sup
        v_lim = v_lim_inf, se v < v_lim_inf
        
    Inputs:
        -> rede = sistema elétrico de testes
        -> matriz_G = matriz de condutância nodal do sistema
        -> v_lim_sup = vetor contendo os limites superiores de tensão nas barras do sistema
        -> v_lim_inf = vetor contendo os limites inferiores de tensão nas barras do sistema
    Outputs:
        -> f = função objetivo do problema de FPOR
        -> pen_v = penalidade de violação dos limites de tensão das barras
    
    """
    
    v_k = np.array([rede.res_bus.vm_pu.to_numpy(dtype = np.float64)])
    v_m = v_k.T
    
    theta_k = np.radians(np.array([rede.res_bus.va_degree.to_numpy(dtype = np.float64)]))
    theta_m = theta_k.T
    
    #Calculo da função objetivo
    f = np.power(v_k,2) + np.power(v_m,2) - 2*np.multiply(np.multiply(v_k,v_m),np.cos(theta_k-theta_m))
    f = np.multiply(matriz_G, f)
    f=np.squeeze(np.sum(np.array([f[np.nonzero(f)]])))
    
    #Calculo da penalidade das tensões
    
    #Violação do limite superior (v - v_lim), v > v_lim_sup
    v_up = v_k - v_lim_sup
    v_up = v_up[np.greater(v_up, 0.0)]
    
    #Violação do limite inferior (v-v_lim), v < v_lim_inf
    v_down = v_k - v_lim_inf
    v_down = v_down[np.less(v_down, 0.0)]
    
    pen_v = np.squeeze(np.sum(np.square(v_up)) + np.sum(np.square(v_down)))

    return f, pen_v


def penalidade_senoidal_tap(alcateia = None, DEBUG = False, vec_debug = None):
    '''
    Esta função retorna a penalidade senoidal sobre os taps dos transformadores para toda a alcateia.
    Dado um tap t e um passo discreto s, a função penalidade senoidal é dada por:
        pen_sen_tap = sum {sen^2(t*pi/s)}    
    
    Inputs:
        -> alcateia
        
    Outputs:
        -> pen_taps: um vetor cuja forma é (nt, n_lobos) contendo as penalizações referentes aos taps para toda a alcateia
    '''
    if DEBUG:
        taps = vec_debug
        pen_taps = np.zeros(shape = vec_debug.shape)
    else:
        #Variável para receber os taps de todos os lobos
        taps = alcateia[ng:ng+nt, :]
        #Variável para armazenar as penalidades referentes aos taps de todos os lobos
        pen_taps = np.array(np.zeros((1, alcateia.shape[1])))
    
    #Executa a equação da penalização sem efetuar a soma
    taps = np.square(np.sin(taps*np.pi/tap_step))
    pen_taps = taps
    
    if not DEBUG:
        #Executa a soma ao longo das colunas da variável taps
        pen_taps = np.sum(taps, axis=0)
    
    threshold = np.less_equal(pen_taps, 1e-15)
    pen_taps[threshold] = 0.0

    #Deleta a variável taps
    del taps
    
    return pen_taps

def penalidade_senoidal_shunt(conjunto_shunts, alcateia = None, DEBUG = False, vec_debug = None):
    '''
    Esta função retorna a penalidade senoidal sobre os shunts de toda a 
    alcateia.
    
    Seja 'conjunto' a lista de valores discretos que o shunt admite: 
        conjunto = [b_1, b_2, b_3, b_4].
    Seja b um shunt
    
    A função pen_sen_shunt deve ser nula caso 'b' pertença a 'conjunto' e 
    maior que 0 caso contrário.
    
    Define-se a variável a função pen_sen_shunt para o caso de um único shunt b como:
        
        pen_sen_shunt = sen[ pi * (b /(b_sup - b_inf)) + alfa ]
    
    Onde:
        - b_sup: é o valor discreto superior mais próximo de 'b'
        - b_inf: é o valor discreto inferior mais próximo de 'b'
        - alfa: é uma variável escolhida para que pen_sen_shunt = 0 caso 'b' pertença a 'conjunto'
    
    Alfa é dada por:
        
        alfa = pi*[ ceil{b_inf/(b_sup - b_inf)} - b_inf/(b_sup - b_inf)]
        
        *ceil(x) é o valor de x arredondado para o inteiro superior mais próximo
    
    Inputs:
        -> alcateia 
        -> conjunto_shunts = conjunto de valores que cada shunt de 'alcateia' pode admitir
        
    Outputs:
        -> pen_shunts: um vetor cuja forma é (ns, n_lobos) contendo as penalizações referentes aos shunts para toda a alcateia
    
    '''
    if DEBUG:
        shunts = vec_debug
        alfa = np.zeros(shape=vec_debug.shape)
    else:
        #Variável para receber os shunts da alcateia
        shunts = alcateia[ng+nt:ng+nt+ns, :]
        #A variável alfa será um vetor no formato (ns, 1)
        alfa = np.zeros(shape=(ns, alcateia.shape[-1]))
    
    
    
    #Variáveis temporárias para armazenar b_inf's e b_sup's, obtidos pelas funções auxiliares descritas no ínicio do código
    shunts_sup = discreto_superior(shunts, conjunto_shunts)
    shunts_inf = discreto_inferior(shunts, conjunto_shunts)
    
    d = shunts_sup - shunts_inf
    if DEBUG:
        print('d: {}'.format(d))
    alfa = np.pi * (np.ceil(shunts_inf/d) - shunts_inf/d)
    
    pen_shunts = np.sin(alfa + np.pi*(shunts/d))
    pen_shunts = np.square(pen_shunts)
    if not DEBUG:
        pen_shunts = np.sum(pen_shunts, axis = 0)
    threshold = np.less_equal(pen_shunts, 1e-12)
    pen_shunts[threshold] = 0.0
    
    return pen_shunts

def inicializar_alcateia(n_lobos, rede, parametros_rede):
    def inicializar_alcateia(n_lobos, rede, parametros_rede):
    '''
    Esta função inicializa a alcateia de lobos (agentes de busca) como uma matriz (numpy array) com formato (dim+5, n_lobos)
    
        -> Cada coluna da matriz representa um lobo.
        -> As linhas de 0 a dim-1 são as variáveis do problema.
        -> A linha dim armazena a função objetivo para cada lobo;
        -> A linha dim+1 armazena a penalidade das tensões para cada lobo;
        -> A linha dim+2 armazena a penalidade dos taps para cada lobo;
        -> A linha dim+3 armazena a penalidade dos shunts para cada lobo;  
        -> A linha dim+4 armazena a função fitness para cada lobo;
        
        
    Inputs:
        -> n_lobos = número de agentes de busca;
        -> rede
        -> parametros_rede - obtido via função gerenciar_rede
    
    Output:
        -> Alcateia
    '''
    
    '''
    Variável que armazena o número de variáveis do problema.
    ng = número de barras geradoras do sistema;
    nt = número de transformadores com controle de tap do sistema;
    ns = número de susceptâncias shunt do sistema.
    '''
    dim = ng + nt+ ns
    
    #Inicialização da Alcateia com zeros
    alcateia = np.zeros(shape = (dim+5, n_lobos), dtype = np.float32)
    
    #Inicialização aleatória das variáveis contínuas (tensões das barras geradoras) a partir de uma distribuição normal
    alcateia[:ng, :] = np.random.uniform(rede.bus.min_vm_pu[0], rede.bus.max_vm_pu[1], size=(ng,n_lobos))
    
    #Inicialização dos taps dos transformadores a partir da escolha aleatória dentro dos valores discretos permitidos
    alcateia[ng:ng+nt, :] = np.random.choice(parametros_rede["Valores_taps"], size =(nt, n_lobos))
    
    #Inicialização dos shunts das barras a partir da escolha aleatória dentro dos valores discretos permitidos
    #Não consegui escapar do loop de for aqui ainda =/
    for i in range(ns):
        alcateia[ng+nt+i, :] = np.random.choice(parametros_rede["Valores_shunts"][str(nb)][i], size = (ns,n_lobos))
    
    #Inicializar a função objetivo, as funções de penalização e a função fitness de cada lobo
    alcateia[dim:dim+5, :] = 0.0
    
    
    #Inserir o lobo w_1 com os valores do ponto de operação da rede 
    alcateia[:dim, 0] = parametros_rede["Lobo1"]
    
    return alcateia


def fluxo_de_carga(rede, alcateia, conjunto_shunts):
    '''
    Esta função executa o fluxo de carga para todos os lobos da alcateia utilizando a função 'runpp' da biblioteca
    PandaPower, alterando as posições de todos os lobos para a região factível do problema de FPOR.
    Após executar o fluxo para cada lobo, a função objetivo e as penalidades são calculadas e inseridas na alcateia, que
    depois será otimizada.
    
    Inputs:
        -> alcateia
        -> rede
        -> parametros_rede
    Outputs:
        -> alcateia
    '''
    
    '''
    Variável que armazena o número de variáveis do problema.
    ng = número de barras geradoras do sistema;
    nt = número de transformadores com controle de tap do sistema;
    ns = número de susceptâncias shunt do sistema.
    '''
    dim = ng + nt+ ns
    
    #Loop sobre cada lobo (linha da alcateia transposta) para executar o fluxo de carga
    #Infelizmente enquanto utilizar o PandaPower, é impossível se livrar deste loop
    alcateia_transposta = alcateia.T
    for indice_lobo, lobo in enumerate(alcateia_transposta):
        v_lobo = lobo[:ng]
        taps_lobo = lobo[ng:ng+nt]
        shunts_lobo = lobo[ng+nt:ng+nt+ns]
        
        #Inserindo as tensões das barras de geração na rede
        rede.gen.vm_pu = v_lobo
        
        #Inserindo os taps dos transformadores
        '''
        Os taps dos transformadores devem ser inseridos como valores de posição, 
        e não como seu valor em pu. Para converter de pu para posição é utilizada a seguinte equação:
        
            tap_pos = [(tap_pu - 1)*100]/tap_step_percent] + tap_neutral
    
        O valor tap_mid_pos é 0 no sistema de 14 barras
        '''
        rede.trafo.tap_pos[:nt] = rede.trafo.tap_neutral[:nt] + ((taps_lobo - 1.0)*(100/rede.trafo.tap_step_percent[:nt]))
        
        #Inserindo as susceptâncias shunt
        """
        A unidade de susceptância shunt no pandapower é MVAr e seu sinal é negativo. 
        Para transformar de pu para MVAr negativo basta multiplicar por -100
        """
        rede.shunt.q_mvar = shunts_lobo*(-100)
        
        #Soluciona o fluxo de carga utilizando o algoritmo Newton-Raphson
        pp.runpp(rede, algorithm = 'nr', numba = True, init = 'results')
        
        #Recebendo os valores das tensões das barras, taps e shunts e armazenando no lobo    
        v_lobo = rede.res_gen.vm_pu.to_numpy(dtype = np.float32)

        #Recebendo a posição dos taps e convertendo pra pu
        taps_lobo = 1 + ((rede.trafo.tap_pos[:nt] - rede.trafo.tap_neutral[:nt])*(rede.trafo.tap_step_percent[:nt]/100))
        
        #Recebendo o valor da susceptância shunt e convertendo para pu
        shunts_lobo = rede.res_shunt.q_mvar.to_numpy(dtype = np.float32)/(-100) 
        
        #Atualizando o lobo
        lobo[:ng] = v_lobo
        lobo[ng:ng+nt] = taps_lobo
        lobo[ng+nt:ng+nt+ns] = shunts_lobo
        
        lobo[dim], lobo[dim + 1] = funcao_objetivo_e_pen_v(rede, matriz_G, v_lim_sup, v_lim_inf)
        
        alcateia_transposta[indice_lobo] = lobo
    
    alcateia = alcateia_transposta.T
    
    alcateia[dim + 2, :] = penalidade_senoidal_tap(alcateia = alcateia)
    alcateia[dim + 3, :] = penalidade_senoidal_shunt(conjunto_shunts = conjunto_shunts, alcateia = alcateia)
    alcateia[-1, :] = np.sum(alcateia[dim+1:dim+4, :], axis = 0, keepdims=True)
    
    return alcateia

def otimizar_alcateia(f_obj, pen_v, pen_tap, pen_shunt, lbd, t_max):
    pass

def visualizar_resultados():
    pass


"""
---------------------------------------------------------- TESTES ---------------------------------------------------------
"""

'''
r1 = gerenciar_rede(rede)
pp.runpp(rede, algorithm='fdbx')
alcateia = inicializar_alcateia(12, rede, r1)
'''


