# FPOR_discreto_GWO

Desenvolvimento do algoritmo Grey Wolf Optimizer associado a funções penalidade para a resolução do problema de Fluxo de Potência Ótimo Reativo (FPOR) com variáveis discretas. Os códigos salvos neste repositório fazem parte de uma pesquisa de Iniciação Científica desenvolvida por Leonardo P. A. Campos sob orientação da Drª Edilaine Martins Soler, com financiamento da FAPESP (bolsa de iniciação científica).

Todo o código desenvolvido para a abordagem de solução proposta foi desenvolvido na linguagem Python. As bibliotecas utilizadas são: NumPy (para manipulação de matrizes e cálculo numérico), PandaPower (para gerenciar os sistemas elétricos de teste e solucionar o problema de Fluxo de Carga), Matplotlib (para plotar graficamente os resultados), Time (para contar o tempo de execução do algoritmo) e Tabulate (para gerar tabelas e facilitar a visualização dos resultados).

O notebook "FPORdiscreto_GWO_Demo.ipynb" contém um botão que permite acessá-lo e executá-lo interativamente via Google Colab, de forma que não é necessária uma instalação local da linguagem Python e demais dependências.

# Resumo

O Problema de Fluxo de Potência Ótimo (FPO) determina o estado de uma rede elé-trica de potência para desempenho ótimo, satisfazendo suas restrições físicas e operacionais.  Umcaso particular deste problema é o Fluxo de Potência Ótimo Reativo (FPOR), que pode ser mode-lado matematicamente como um problema de otimização não linear, não-convexo, restrito e comvariáveis discretas e contínuas.  Na maioria dos trabalhos da literatura, as variáveis discretas sãotratadas como contínuas, dada a dificuldade de solucionar o problema.  Neste trabalho é propostauma abordagem de solução para o problema de FPOR considerando a natureza discreta de algumasvariáveis.  Foi utilizada uma função penalidade, associada ao algoritmo meta-heurísticoGrey WolfOptimizer (GWO), para tratar as variáveis discretas do problema.  Testes numéricos feitos com ossistemas benchmark IEEE de 14 e 30 barras demonstraram o potencial da abordagem proposta emobter soluções discretas de boa qualidade.

# Abstract

The Optimal Power Flow (OPF) problem determines the state of an electric power gridfor optimal performance, satisfying its physical and operational constraints.  A particular case ofthat problem is the Optimal Reactive Power Dispatch (ORPD), which can be mathematically mo-delled as a nonlinear, non-convex, constrained, with discrete and continuous variables optimizationproblem.   Most  approaches  in  the  literature  treat  the  discrete  variables  as  continuous,  given  thedifficulty to solve the problem.  This work proposes a solution approach for the ORPD problemconsidering the discrete nature of some of its variables. A penalty function, associated to the GreyWolf Optimizer (GWO) meta-heuristic algorithm, was used to manage the discrete variables.  Nu-merical tests conducted on the 14 and 30 bus IEEE benchmark systems demonstrated the potentialof the proposed approach to obtain good quality discrete solutions.

