// Referências da página do modelo: livros, artigos e dados que embasam o
// Vitrôncio ou que foram testados na pesquisa de novembro de 2026
// (research/2026-11). Os links abrem a busca do título no Google Acadêmico.
// Só entra aqui o que foi lido ou usado; não invente referência.

export const RESEARCH = [
  // [o que foi testado, situação, efeito no RPS 2022–2026 sobre a versão anterior]
  ['Contexto do jogo: time já rebaixado ou campeão, jogo continental logo depois, mando fora da sede, portões fechados', 'entrou (v2)', '−0,0011'],
  ['xG medido chute a chute, como segundo rating de gols', 'entrou (v3)', '−0,0001'],
  ['Cadeia de Markov calibrada para reproduzir os gols esperados de cada lado', 'entrou (v3.1)', '−0,0004'],
  ['Escalação anunciada: time habitual em campo, gols e assistências em campo, banco', 'entrou (v3.2)', '−0,0008'],
  ['Desfalques conhecidos na véspera: suspensos por cartões e titulares habituais fora do último jogo', 'entrou (v3.3), só enquanto a escalação não sai', '−0,0005 (sem escalação)'],
  ['Árvores de decisão com atributos de elenco', 'ganho igual ao da escalação; ficou a versão linear, que roda no navegador', '≈ −0,0008'],
  ['Copa do Brasil, Libertadores e estaduais como evidência de força', 'não entrou', '≈ 0'],
  ['Descanso, sequência de jogos, viagem, clima, clássicos, pressão contra o rebaixamento', 'não entrou', '≈ 0'],
  ['Mudança de técnico (Wikipedia): primeiro jogo após a saída do técnico e técnico interino', 'não entrou: ganho pequeno e dado frágil (limite de acesso, interinos desatualizados)', '−0,0003'],
  ['Múltiplas sugeridas pelo Best Against (2 a 5 pernas por rodada)', 'na aba Múltiplas, com o retrospecto', '−11% (2 pernas), −50% (3) por aposta'],
  ['Árbitro: tendência a mando e gols', 'não entrou: a ESPN quase não tem o árbitro de 2014 a 2021', '≈ 0'],
  ['Pi-ratings, Elo, GAP e Berrar combinados por árvores', 'não entrou: o ganho era só recalibração', '≈ 0'],
  ['Redes neurais sobre o histórico de cada time (DeepSets, GRU)', 'não entrou', '−0,0001 (ruído)'],
  ['Novo canal de gols com covariância ataque–defesa; rating ordinal de vitória/empate/derrota', 'não entrou', '+0,0001'],
];

const R = (authors, year, title, venue, use) => ({ authors, year, title, venue, use });
export const REFS = [
  ['Força dos times e modelos de placar', [
    R('Maher, M. J.', 1982, 'Modelling association football scores', 'Statistica Neerlandica 36(3), 109–118', 'Gols de cada lado como Poisson com ataque e defesa'),
    R('Dixon, M. J.; Coles, S. G.', 1997, 'Modelling association football scores and inefficiencies in the football betting market', 'Journal of the Royal Statistical Society C 46(2), 265–280', 'Forças que variam no tempo; comparação com o mercado'),
    R('Rue, H.; Salvesen, Ø.', 2000, 'Prediction and retrospective analysis of soccer matches in a league', 'Journal of the Royal Statistical Society D 49(3), 399–418', 'Forças dinâmicas com incerteza'),
    R('Karlis, D.; Ntzoufras, I.', 2003, 'Analysis of sports data by using bivariate Poisson models', 'Journal of the Royal Statistical Society D 52(3), 381–393', 'Dependência entre os gols dos dois lados'),
    R('Glickman, M. E.; Stern, H. S.', 1998, 'A state-space model for National Football League scores', 'Journal of the American Statistical Association 93(441), 25–35', 'Força como estado que evolui jogo a jogo'),
    R('Koopman, S. J.; Lit, R.', 2015, 'A dynamic bivariate Poisson model for analysing and forecasting match results in the English Premier League', 'Journal of the Royal Statistical Society A 178(1), 167–186', 'Espaço de estados com Poisson'),
    R('Koopman, S. J.; Lit, R.', 2019, 'Forecasting football match results in national league competitions using score-driven time series models', 'International Journal of Forecasting 35(2), 797–809', 'Modelo GAS (testado)'),
    R('Kalman, R. E.', 1960, 'A new approach to linear filtering and prediction problems', 'Journal of Basic Engineering 82(1), 35–45', 'Base do filtro que atualiza as forças'),
    R('Durbin, J.; Koopman, S. J.', 2012, 'Time Series Analysis by State Space Methods', 'Oxford University Press, 2ª ed. (livro)', 'Filtro de Kalman estendido'),
    R('Goddard, J.', 2005, 'Regression models for forecasting goals and match results in association football', 'International Journal of Forecasting 21(2), 331–340', 'Modelos de gols contra modelos de resultado (ordinal testado)'),
    R('Shmueli, G.; Minka, T. P.; Kadane, J. B.; Borle, S.; Boatwright, P.', 2005, 'A useful distribution for fitting discrete data: revival of the Conway–Maxwell–Poisson distribution', 'Journal of the Royal Statistical Society C 54(1), 127–142', 'Distribuição alternativa de gols (testada)'),
  ]],
  ['Dinâmica da partida', [
    R('Dixon, M. J.; Robinson, M. E.', 1998, 'A birth process model for association football matches', 'Journal of the Royal Statistical Society D 47(3), 523–538', 'Chance de gol que depende do minuto e do placar: base da cadeia minuto a minuto'),
    R('Titman, A. C.; Costain, D. A.; Ridall, P. G.; Gregory, K.', 2015, 'Joint modelling of goals and bookings in association football', 'Journal of the Royal Statistical Society A 178(3), 659–683', 'Gols e cartões no mesmo modelo'),
    R('Vecer, J.; Kopriva, F.; Ichiba, T.', 2009, 'Estimating the effect of the red card in soccer', 'Journal of Quantitative Analysis in Sports 5(1)', 'Efeito da expulsão'),
  ]],
  ['Vantagem de casa e contexto', [
    R('Pollard, R.', 2006, 'Worldwide regional variations in home advantage in association football', 'Journal of Sports Sciences 24(3), 231–240', 'Mando no Brasil acima da média mundial'),
    R('Pollard, R.; da Silva, C. D.; Medeiros, N. C.', 2008, 'Home advantage in football in Brazil: differences between teams and the effects of distance traveled', 'Revista Brasileira de Futebol 1(1)', 'Distância de viagem (testada)'),
    R('Fischer, K.; Haucap, J.', 2021, 'Does crowd support drive the home advantage in professional football? Evidence from German ghost games during the COVID-19 pandemic', 'Journal of Sports Economics 22(8), 982–1008', 'Portões fechados'),
    R('McCarrick, D.; Bilalic, M.; Neave, N.; Wolfson, S.', 2021, 'Home advantage during the COVID-19 pandemic: analyses of European football leagues', 'Psychology of Sport and Exercise 56, 102013', 'Portões fechados'),
    R('Dupont, G.; Nedelec, M.; McCall, A.; McCormack, D.; Berthoin, S.; Wisløff, U.', 2010, 'Effect of 2 soccer matches in a week on physical performance and injury rate', 'The American Journal of Sports Medicine 38(9), 1752–1758', 'Sequência de jogos e descanso (testados)'),
  ]],
  ['Jogadores e escalação', [
    R('Kharrat, T.; McHale, I. G.; Peña, J. L.', 2020, 'Plus–minus player ratings for soccer', 'European Journal of Operational Research 283(2), 726–736', 'Valor de cada jogador (testado)'),
    R('Hvattum, L. M.', 2019, 'A comprehensive review of plus-minus ratings for evaluating individual players in team sports', 'International Journal of Computer Science in Sport 18(1), 1–23', 'Ratings de jogadores'),
  ]],
  ['Expected goals (xG)', [
    R('Rathke, A.', 2017, 'An examination of expected goals and shot efficiency in soccer', 'Journal of Human Sport and Exercise 12(2)', 'Valor de cada finalização pela posição'),
    R('Mead, J.; O’Hare, A.; McMenemy, P.', 2023, 'Expected goals in football: improving model performance and demonstrating value', 'PLoS ONE 18(4), e0282295', 'Modelo de xG por chute'),
  ]],
  ['Outros ratings (testados)', [
    R('Hvattum, L. M.; Arntzen, H.', 2010, 'Using ELO ratings for match result prediction in association football', 'International Journal of Forecasting 26(3), 460–470', 'Elo'),
    R('Constantinou, A. C.; Fenton, N. E.', 2013, 'Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries', 'Journal of Quantitative Analysis in Sports 9(1), 37–50', 'Pi-ratings'),
    R('Wheatcroft, E.', 2020, 'A profitable model for predicting the over/under market in football', 'International Journal of Forecasting 36(3), 916–932', 'GAP ratings'),
    R('Berrar, D.; Lopes, P.; Dubitzky, W.', 2019, 'Incorporating domain knowledge in machine learning for soccer outcome prediction', 'Machine Learning 108(1), 97–126', 'Ratings de Berrar'),
  ]],
  ['Avaliação, calibração e combinação de previsões', [
    R('Epstein, E. S.', 1969, 'A scoring system for probability forecasts of ranked categories', 'Journal of Applied Meteorology 8(6), 985–987', 'Origem do RPS'),
    R('Constantinou, A. C.; Fenton, N. E.', 2012, 'Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models', 'Journal of Quantitative Analysis in Sports 8(1)', 'RPS para futebol'),
    R('Wheatcroft, E.', 2021, 'Evaluating probabilistic forecasts of football matches: the case against the ranked probability score', 'Journal of Quantitative Analysis in Sports 17', 'Por que também olhamos o log-loss'),
    R('Genest, C.; Zidek, J. V.', 1986, 'Combining probability distributions: a critique and an annotated bibliography', 'Statistical Science 1(1), 114–135', 'Combinação de previsões'),
    R('Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q.', 2017, 'On calibration of modern neural networks', 'ICML 2017', 'Calibração por temperatura (testada)'),
    R('Kull, M.; Perello-Nieto, M.; Kängsepp, M.; Silva Filho, T.; Song, H.; Flach, P.', 2019, 'Beyond temperature scaling: obtaining well-calibrated multi-class probabilities with Dirichlet calibration', 'NeurIPS 2019', 'Calibração de Dirichlet (testada)'),
  ]],
  ['Aprendizado de máquina (testado)', [
    R('Dubitzky, W.; Lopes, P.; Davis, J.; Berrar, D.', 2019, 'The Open International Soccer Database for machine learning', 'Machine Learning 108(1), 9–28', 'Desafio de previsão de futebol de 2017'),
    R('Hubáček, O.; Šourek, G.; Železný, F.', 2019, 'Learning to predict soccer results from relational data with gradient boosted trees', 'Machine Learning 108(1), 29–47', 'Árvores sobre ratings'),
    R('Bunker, R.; Susnjak, T.', 2022, 'The application of machine learning techniques for predicting match results in team sport: a review', 'Journal of Artificial Intelligence Research 73, 1285–1322', 'Panorama'),
    R('Ke, G. et al.', 2017, 'LightGBM: a highly efficient gradient boosting decision tree', 'NeurIPS 2017', 'Árvores usadas nos testes'),
    R('Zaheer, M.; Kottur, S.; Ravanbakhsh, S.; Póczos, B.; Salakhutdinov, R.; Smola, A.', 2017, 'Deep Sets', 'NeurIPS 2017', 'Rede neural testada'),
    R('Cho, K. et al.', 2014, 'Learning phrase representations using RNN encoder–decoder for statistical machine translation', 'EMNLP 2014', 'GRU, rede neural testada'),
  ]],
  ['Mercado de apostas (só como régua de comparação)', [
    R('Forrest, D.; Goddard, J.; Simmons, R.', 2005, 'Odds-setters as forecasters: the case of English football', 'International Journal of Forecasting 21(3), 551–564', 'Por que o mercado é difícil de bater'),
    R('Spann, M.; Skiera, B.', 2009, 'Sports forecasting: a comparison of the forecast accuracy of prediction markets, betting odds and tipsters', 'Journal of Forecasting 28(1), 55–72', 'Odds como previsão'),
    R('Angelini, G.; De Angelis, L.', 2019, 'Efficiency of online football betting markets', 'International Journal of Forecasting 35(2), 712–721', 'Eficiência do mercado'),
    R('Cain, M.; Law, D.; Peel, D.', 2000, 'The favourite-longshot bias and market efficiency in UK football betting', 'Scottish Journal of Political Economy 47(1), 25–36', 'Viés favorito-azarão'),
    R('Shin, H. S.', 1993, 'Measuring the incidence of insider trading in a market for state-contingent claims', 'The Economic Journal 103(420), 1141–1153', 'Retirar a margem das odds'),
    R('Štrumbelj, E.', 2014, 'On determining probability forecasts from betting odds', 'International Journal of Forecasting 30(4), 934–943', 'Retirar a margem das odds'),
    R('Hubáček, O.; Šourek, G.; Železný, F.', 2019, 'Exploiting sports-betting market using machine learning', 'International Journal of Forecasting 35(2), 783–796', 'Modelo independente do mercado'),
    R('Kaunitz, L.; Zhong, S.; Kreiner, J.', 2017, 'Beating the bookies with their own numbers — and how the online sports betting market is rigged', 'arXiv:1710.02824', 'Casas limitam quem ganha'),
  ]],
  ['Livros', [
    R('Anderson, C.; Sally, D.', 2013, 'The Numbers Game: Why Everything You Know About Football Is Wrong', 'Penguin', 'Gols como evento raro e sorte no futebol'),
    R('Sumpter, D.', 2016, 'Soccermatics: Mathematical Adventures in the Beautiful Game', 'Bloomsbury Sigma', 'Modelos de jogo'),
    R('Kuper, S.; Szymanski, S.', 2009, 'Soccernomics', 'Nation Books', 'Economia e dados no futebol'),
    R('Tippett, J.', 2019, 'The Expected Goals Philosophy', 'edição do autor', 'xG'),
    R('Biermann, C.', 2019, 'Football Hackers: The Science and Art of a Data Revolution', 'Blink Publishing', 'Dados no futebol'),
    R('Albert, J.; Koning, R. H. (orgs.)', 2007, 'Statistical Thinking in Sports', 'Chapman & Hall/CRC', 'Estatística no esporte'),
  ]],
];

export const DATA = [
  ['ESPN', 'placares, lances, estatísticas, escalações e finalizações (API pública), Séries A e B desde 2006 e taças continentais'],
  ['football-data.co.uk', 'odds de fechamento da Série A desde 2012, só para comparação'],
];
