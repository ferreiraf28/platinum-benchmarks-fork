# Project Proposal
## Selective Human Review for Scalable Platinum Benchmark Curation

### 1. Resumo executivo

Este projeto investiga duas limitações relacionadas explicitamente reconhecidas
em *Do Large Language Model Benchmarks Test Reliability?* (Vendrow et al.,
2025): a dependência de inspeção humana sempre que os modelos falham ou
discordam, que limita a escalabilidade, e a possibilidade de perguntas
ambíguas passarem quando todos os modelos produzem a mesma resposta.

Propomos uma extensão pequena, reprodutível e executável num mês por três
estudantes:

1. **Auditor seletivo offline (contribuição principal):** usar apenas as
   respostas já gravadas na cache pública do artigo para estimar a confiança
   observável de cada pergunta e decidir quais podem ser aceites
   automaticamente e quais devem ser enviadas para revisão humana.
2. **Stress test controlado (experimento secundário):** testar uma amostra
   pequena dos casos aceites pelo auditor para medir quanto risco de consenso
   correlacionado permanece escondido.

O Arm 1 é totalmente offline e suficiente para a contribuição principal. O
stress test não é uma segunda solução simétrica: é uma análise de robustez e
ameaça à validade. O projeto não promete eliminar a curadoria humana; promete
medir se consegue reduzir substancialmente o número de inspeções mantendo um
risco controlado nos exemplos aceites.

### 2. Alinhamento com o enunciado

O trabalho segue diretamente as três fases pedidas no enunciado:

- **Análise da contribuição original:** reproduzir a lógica de avaliação e
  estudar a limitação do consenso entre modelos.
- **Proposta alternativa:** introduzir um auditor seletivo pós-hoc que calcula
  incerteza a partir das respostas em cache e decide entre aceitar
  automaticamente e deferir para revisão humana.
- **Avaliação empírica:** comparar o método original com o auditor no mesmo
  conjunto de benchmarks, usando os rótulos `platinum_target` como referência
  de correção e reportando cobertura, risco e área sob a curva risco-cobertura.

O resultado é uma alternativa de **triagem e priorização**. A comparação
principal será entre revisão total, maioria simples e auditor seletivo,
medindo quantas inspeções são poupadas e qual o risco introduzido.

### 3. Problema e limitação estudada

O repositório oficial implementa a avaliação em `src/run_benchmark.py` e
carrega respostas de ficheiros Pickle através de `src/utils.py` e
`src/models.py`. A cache contém respostas textuais indexadas por
`(prompt, temperature, run_id, model_name)`, mas não contém um campo separado
para consenso nem uma decisão humana de “pergunta segura”.

O artigo reconhece que a sua estratégia pode falhar quando uma pergunta
ambígua é interpretada da mesma maneira por todos os modelos. A limitação
principal escolhida é a falta de escalabilidade da inspeção manual. A hipótese
é que a combinação de sinais observáveis na cache identifica casos de maior
risco e os encaminha para revisão, reduzindo o trabalho humano sem aceitar
indiscriminadamente a maioria.

### 4. Perguntas de investigação

**RQ1.** A discordância entre modelos, calculada sobre as respostas extraídas
da cache, identifica instâncias que devem ser priorizadas para revisão humana?

**RQ2.** Um score simples que combine entropia dos votos e sinais de
variabilidade textual permite trocar cobertura por menor risco de forma
previsível?

**RQ3.** Entre as instâncias aceites automaticamente, pequenas perturbações
semânticas invariantes revelam fragilidade num subconjunto controlado?

**RQ4.** Quais são as limitações práticas da cache e do procedimento original
para auditoria automática sem chamadas a APIs?

### 5. Hipóteses testáveis

- **H1:** instâncias com maior entropia de votos têm maior taxa de erro do
  ensemble relativamente ao `platinum_target`.
- **H2:** um score combinado de força da maioria, entropia e estabilidade
  textual produz uma curva risco-cobertura melhor do que a confiança baseada
  apenas na maioria e reduz a carga de revisão para um risco definido.
- **H3:** pelo menos uma classe de perturbação controlada produz uma taxa de
  quebra de consenso superior à de uma transformação placebo.

H3 será considerada exploratória. Se o modelo local não estiver disponível ou
se a perturbação não puder preservar a resposta com confiança, o resultado
válido será documentar essa limitação e concluir apenas os braços offline.

### 6. Dados e reprodutibilidade

#### 6.1 Fonte dos dados

- Dataset: `madrylab/platinum-bench-paper-version`, para alinhar com o artigo.
- Cache: `madrylab/platinum-bench-paper-cache`, descarregada por
  `scripts/download_paper_cache.sh`.
- Código reutilizado:
    - `src/utils.py`: `LLMCache`, `get_llm_cache`, `get_parse_fn`,
    `check_prediction` e `get_prompt`;
    - `src/models.py`: construção das chaves da cache e
    `ModelInferenceEngine.run_inference`;
    - `src/run_benchmark.py`: lista de datasets, filtragem de rejeitados e
    campos do benchmark;
    - `scripts/get_paper_results.sh`: lista histórica de modelos.

#### 6.2 Regra de execução sem rede

O notebook não chamará `run_benchmark` com o comportamento predefinido,
porque esse caminho pode fazer uma chamada remota quando falta uma entrada.
Será criado um loader de análise que:

1. abre os `.pkl` em modo leitura;
2. constrói as chaves exatamente como o repositório;
3. lança um erro explícito para cache misses;
4. nunca instancia clientes OpenAI, Anthropic, Google ou outros;
5. guarda um relatório de cobertura da cache.

Todas as tabelas e gráficos do Arm 1 serão gerados sem API e sem GPU.

#### 6.3 Escopo de datasets

Para garantir profundidade e conclusão no prazo:

- **Desenvolvimento:** `gsm8k`, `svamp` e `drop`;
- **Avaliação principal:** todos os datasets textuais disponíveis na versão do
  artigo, se a cobertura da cache for suficiente;
- **Fallback obrigatório:** os três datasets de desenvolvimento, com o mesmo
  protocolo e limitações claramente reportados;
- `vqa` fica fora do núcleo por depender de imagens e por aumentar
  desnecessariamente a complexidade de alinhamento.

Antes de qualquer análise será produzido um inventário com número de exemplos,
modelos e cache misses por dataset.

### 7. Metodologia

## Arm 1 — Auditoria seletiva offline (resultado principal)

#### 7.1 Construção da matriz de respostas

Para cada pergunta e cada modelo:

1. localizar a resposta textual na cache;
2. selecionar o prompt correto com `get_prompt`;
3. extrair a resposta final com `get_parse_fn`;
4. guardar a resposta textual original para análise de justificação;
5. guardar `platinum_target` apenas numa tabela de avaliação, nunca como
   feature do score.

O produto será uma tabela em formato Parquet ou CSV, gerada pelo notebook,
com os campos:

```text
dataset, example_id, prompt, model, raw_response,
prediction, cleaning_status, cache_key, extraction_status
```

O Parquet/CSV é apenas um artefacto derivado para análise; a fonte original
continua a ser o Pickle oficial. Uma tabela de avaliação separada acrescentará
`platinum_target` e `is_correct`.

#### 7.2 Sinais de incerteza

Para cada pergunta serão calculados:

- **entropia de votos:** entropia de Shannon da distribuição das respostas
  extraídas;
- **majority ratio:** frequência relativa da resposta mais votada;
- **número de respostas distintas;**
- **comprimento da resposta:** número de caracteres e palavras;
- **similaridade das justificações:** média por pares usando TF-IDF e cosseno;
- **estabilidade do comprimento:** transformação normalizada do coeficiente de
  variação dos textos;
- **marcadores de hesitação:** feature exploratória com termos como `however`,
  `assuming`, `unclear` e `cannot determine`.

Os sinais textuais serão apresentados como indicadores exploratórios, não como
prova de que o modelo expôs um raciocínio fiel. O CoT será tratado como texto
observável da resposta e não como ground truth do processo interno.

#### 7.3 Separação entre score e ground truth

O `platinum_target` **não entra no score de decisão**. Se entrasse, o auditor
teria acesso à resposta correta e o experimento sofreria de *label leakage*.
Durante a decisão só são usadas respostas finais extraídas, distribuição de
votos, textos brutos, justificações, características de estabilidade textual e
completude da cache.

O `platinum_target` entra apenas para calcular `is_correct` retrospectivamente,
calibrar o limiar no desenvolvimento, calcular risco/cobertura/AURC na
avaliação e identificar falsos negativos.

#### 7.4 Cálculo do score

Será comparada uma família pequena de scores, evitando treino desnecessário:

- `majority_confidence = majority_ratio`;
- `vote_confidence = 1 - normalized_vote_entropy`;
- `combined_confidence`, combinação normalizada de maioria, entropia,
  similaridade e estabilidade textual.

Para K modelos, a entropia dos votos é:

> **H(x) = −Σ p(c | x) × log p(c | x)**

A confiança dos votos é a entropia normalizada e invertida:

> **VoteConfidence(x) = 1 − H(x) / log(K)**

Se Rᵢ for a justificação do modelo i, a similaridade textual é a média da
similaridade coseno entre todos os pares de vetores TF-IDF:

> **TextSimilarity(x) = média [cos(TFIDF(Rᵢ), TFIDF(Rⱼ))] para todos os pares i < j**

O baseline transparente do score é:

> **S(x) = 0,40 × MajorityRatio(x) + 0,35 × VoteConfidence(x) + 0,15 × TextSimilarity(x) + 0,10 × LengthStability(x)**

Os pesos são um baseline explicável, não uma constante científica. Uma versão
calibrada será comparada usando apenas o conjunto de desenvolvimento.

Os limiares serão determinados apenas no conjunto de desenvolvimento. A
avaliação final usará um split por exemplos, fixado e documentado antes dos
resultados.

Uma instância é:

- **aceite automaticamente** quando S(x) ≥ τ;
- **deferida para revisão humana** quando S(x) < τ;
- **deferida obrigatoriamente** quando há cache miss ou respostas insuficientes.

“Deferida” significa “enviar para revisão adicional”; não significa que a
pergunta está errada. Esta distinção evita transformar o reject option num
classificador artificial de perguntas más.

Exemplo operacional para cinco modelos:

```text
29, 29, 29, 29, 29 -> score alto; aceitar, sujeito ao limiar
29, 29, 29, 29, 36 -> score intermédio; aceitar ou deferir
29, 29, 29, 36, 22 -> score baixo; deferir
36, 36, 36, 36, 36 -> score alto, mesmo que esteja errado
```

O último caso é um falso negativo potencial: consenso não é prova de correção
e motiva o stress test secundário.

#### 7.5 Métricas

Serão reportados:

- risco seletivo: taxa de erro entre as instâncias aceites;
- cobertura: proporção de instâncias aceites;
- redução de revisão: 1 − fração deferida;
- curva risco-cobertura;
- AURC;
- precisão e recall dos erros entre os casos deferidos;
- risco a coberturas alvo, por exemplo 50%, 75% e 90%;
- comparação com maioria simples e seleção aleatória;
- resultados separados por dataset e agregados.

O baseline obrigatório é a maioria simples. Um segundo baseline é a seleção
aleatória repetida com sementes fixas. Para a mesma cobertura, será comparado
o risco do auditor com os dois baselines.

## Arm 2 — Stress test controlado (resultado secundário)

#### 7.5 Definição operacional de consenso

Como o repositório não fornece necessariamente um rótulo `consensus`, uma
instância consensual será definida como:

```text
todos os modelos disponíveis produzem a mesma resposta extraída
e essa resposta coincide com platinum_target
e cleaning_status != rejected
```

O notebook guardará também o número de modelos disponíveis. Instâncias com
cache incompleta não serão chamadas consensuais.

#### 7.6 Subconjunto e perturbações

O stress test será executado apenas em `drop`, onde o artigo documenta o
fenómeno de *first-event bias*, e no máximo em vinte instâncias consensuais
selecionadas antes da análise final.

Serão implementadas duas transformações:

1. **Temporal controlada:** trocar uma pergunta “o que aconteceu primeiro?” por
   “o que aconteceu segundo?”, apenas quando o contexto contém dois eventos
   identificáveis e o novo alvo puder ser verificado manualmente.
2. **Placebo textual:** inserir uma frase neutra que não altera o contexto
   factual nem a resposta.

Cada transformação terá uma função determinística, um teste unitário e um
registo do alvo esperado. Perguntas que não satisfizerem as pré-condições serão
excluídas, não alteradas heuristicamente.

#### 7.7 Inferência e métrica

As perguntas perturbadas não existem na cache original. Para evitar uma
promessa incompatível com o repositório, o plano principal é usar um único
modelo local leve, com execução documentada e seeds fixas, ou uma cache
adicional pequena produzida previamente pela equipa.

Se forem usados modelos locais, os resultados serão descritos como um
**estudo de sensibilidade**, não como reprodução dos modelos frontier do
artigo. O resultado é a taxa de instâncias para as quais a resposta do modelo
local muda ou deixa de coincidir com o consenso original, comparada com o
placebo. A métrica será chamada `consensus-break rate` no notebook para não
confundi-la com uma propriedade observada nos modelos do artigo.

### 8. Desenho experimental

1. Congelar versões do código, dataset, cache, modelos e sementes.
2. Fazer inventário da cache e documentar cobertura.
3. Construir a matriz de respostas.
4. Separar desenvolvimento e avaliação sem misturar limiares e resultados.
5. Calibrar os scores no desenvolvimento.
6. Calcular risco-cobertura e AURC na avaliação.
7. Selecionar o subconjunto consensual para o Arm 2 sem usar os resultados das
   perturbações.
8. Executar as duas perturbações e o placebo.
9. Fazer análise de casos, incluindo falsos negativos e exemplos deferidos.
10. Registar todas as limitações e falhas de cobertura.

### 9. Divisão de trabalho para três estudantes

**Estudante A — Dados e reprodutibilidade**

- inventário dos `.pkl`;
- loader offline;
- integração com o dataset;
- testes de parsing e cache misses;
- geração da matriz de respostas.

**Estudante B — Incerteza e avaliação**

- entropia, majority ratio e features textuais;
- implementação do auditor seletivo;
- risco-cobertura, AURC e baselines;
- tabelas e gráficos.

**Estudante C — Stress test e análise**

- definição e validação dos filtros de consenso;
- operadores temporal e placebo;
- execução do subconjunto local;
- análise qualitativa de casos e limitações.

Todos os estudantes participam na revisão da literatura, interpretação dos
resultados, notebook final e apresentação.

### 10. Cronograma de quatro semanas

#### Semana 1 — Checkpoint 1 e reprodutibilidade

- finalizar a leitura do artigo e da literatura sobre selective prediction,
  reject option e risk-coverage;
- confirmar o problema, o scope e os três datasets;
- executar o inventário da cache;
- gerar uma primeira tabela de respostas;
- preparar a apresentação de 3 minutos para o Checkpoint 1.

**Entregável interno:** notebook mínimo que lê uma cache em modo offline e
produz uma tabela.

#### Semana 2 — Checkpoint 2 e Arm 1

- finalizar a matriz de respostas;
- implementar parsers e validações;
- calcular discordância, entropia e majority ratio;
- implementar os três scores e os baselines;
- preparar problema, limitação, literatura e plano alternativo para a
  apresentação de 5 minutos do Checkpoint 2.

**Critério de saída:** resultados preliminares de risco-cobertura em pelo menos
`gsm8k`, `svamp` e `drop`.

#### Semana 3 — Avaliação e Arm 2

- congelar o protocolo do Arm 1;
- executar a avaliação final e análise de erros;
- implementar e testar as duas perturbações;
- executar o subconjunto de consenso e o placebo;
- analisar se o ambiente local suporta o modelo escolhido.

**Critério de saída:** todos os gráficos principais ou, se o modelo local não
for viável, uma justificação documentada para limitar o Arm 2 a análise
offline e a um protocolo de geração reproduzível.

#### Semana 4 — Consolidação

- repetir os resultados a partir de um ambiente limpo;
- limpar o notebook para execução de início ao fim;
- consolidar tabelas, limitações e ameaças à validade;
- preparar slides, anexos e guião da apresentação;
- confirmar que o pacote final contém todos os ficheiros necessários.

### 11. Entregáveis finais

O ficheiro comprimido conterá:

- notebook pronto a executar;
- módulos Python auxiliares;
- `requirements.txt` ou instruções de instalação;
- README com instruções offline e origem dos dados;
- cache derivada ou amostras permitidas, sem incluir segredos;
- slides em PDF;
- tabelas e figuras reproduzíveis.

Os slides terão no máximo 40 páginas, organizados aproximadamente assim:

1. problema, motivação e artigo base;
2. limitação e revisão de literatura;
3. dados e arquitetura do auditor;
4. desenho experimental;
5. resultados do Arm 1;
6. stress test do Arm 2;
7. análise de casos e limitações;
8. conclusão e trabalho futuro.

### 12. Critérios de sucesso

O projeto será considerado concluído se:

- o notebook reconstruir a matriz de respostas sem chamadas de API;
- a cobertura da cache for reportada explicitamente;
- a discordância e a entropia forem verificadas em exemplos manuais;
- existir comparação quantitativa entre maioria, aleatório e auditor seletivo;
- forem apresentadas curvas risco-cobertura e AURC;
- o Arm 2 tiver pelo menos uma perturbação válida e um placebo, ou uma
  justificação reprodutível para a sua limitação;
- todos os resultados puderem ser regenerados a partir das instruções;
- as conclusões distinguirem claramente evidência, hipótese e limitação.

### 13. Riscos e mitigação

| Risco | Mitigação |
| --- | --- |
| Cache incompleta ou indisponível | começar pelo inventário; fixar três datasets fallback |
| `cleaning_status` não contém `consensus` | inferir consenso pela unanimidade observada e documentar a regra |
| CoT não separado em todos os modelos | tratar como texto bruto e usar features textuais exploratórias |
| `run_benchmark` fazer chamadas de API | usar loader offline próprio e falhar em cache miss |
| Modelo local demasiado pesado | reduzir a vinte exemplos, usar um modelo leve ou reportar Arm 2 como sensibilidade |
| Perturbação altera a resposta correta | exigir pré-condições, revisão manual e placebo |
| Amostras pequenas | reportar intervalos de confiança e evitar conclusões universais |
| Score não melhora o baseline | manter o resultado negativo e analisar por que a limitação não foi resolvida |

### 14. Contribuição esperada

A contribuição é uma avaliação rigorosa e reprodutível de uma extensão de
auditoria seletiva sobre o pipeline real de Platinum Benchmarks. O trabalho:

- reutiliza a contribuição original em vez de a reimplementar;
- transforma a limitação de consenso numa pergunta mensurável;
- mantém o núcleo sem custo de API nem GPU;
- compara uma alternativa com baselines claros;
- inclui um stress test pequeno, controlado e honesto;
- produz um artefacto adequado a um projeto de mestrado de um mês.

O resultado mais forte esperado não é provar que um score simples substitui
revisores humanos. É medir com clareza quanto a cache permite automatizar,
quanto risco permanece nos casos consensuais e em que condições uma auditoria
seletiva é útil.
