# 🍄 Mycelium

**Mycelium** é um motor de busca de texto completo, ultrarrápido e multithreaded, construído em Rust. Ele foi projetado desde o início para performance, aproveitando ao máximo o poder do hardware moderno para indexar e pesquisar grandes volumes de documentos de texto em velocidades impressionantes.

O nome "Mycelium" é inspirado na rede subterrânea dos fungos, que se espalha de forma vasta e eficiente para coletar nutrientes – assim como nosso motor indexa e recupera informações de maneira distribuída e veloz.

## ✨ Visão Geral e Funcionalidades

- **⚡️ Indexação Paralela:** Utiliza a biblioteca `rayon` para paralelizar o processo de indexação, dividindo o trabalho entre todos os núcleos de CPU disponíveis para uma performance massiva.
    
- **🚀 Busca Otimizada:** As buscas também são executadas em paralelo sobre as partições ("shards") do índice. O uso de `AHash`, um hasher não-criptográfico extremamente rápido, acelera as consultas em `HashMap`.
    
- **🧠 Ranking Inteligente:** Os resultados não são apenas listados; eles são classificados por relevância usando o robusto algoritmo **Okapi BM25**. Além disso, uma pontuação de **proximidade** é aplicada para impulsionar documentos onde os termos da busca aparecem mais próximos uns dos outros.
    
- **💾 Compressão Eficiente:** O índice gerado é comprimido com o algoritmo Zstandard (`zstd`), resultando em arquivos de índice significativamente menores no disco sem sacrificar a velocidade de carregamento.
    
- **🛠️ Pipeline de Processamento de Texto:**
    
    - **Tokenização com Regex:** Divide o texto em tokens de forma eficiente.
        
    - **Stemming:** Reduz as palavras às suas raízes (radicais) usando `rust-stemmers` (ex: "running", "ran" -> "run"), melhorando a qualidade da busca.
        
    - **Remoção de Stop Words:** Ignora palavras comuns (como "o", "a", "de") para manter o índice enxuto e relevante.
        
- **👨‍💻 Interface de Linha de Comando (CLI) Amigável:** Construída com `clap`, a CLI é intuitiva, com subcomandos claros para `index` e `search`, barras de progresso informativas e logging configurável.
    

## ⚙️ Como Funciona: Um Mergulho Técnico

Mycelium implementa um **índice invertido particionado (sharded inverted index)**, uma estrutura de dados fundamental para motores de busca modernos.

### 1. A Arquitetura Paralela (Sharding)

No momento da indexação, o Mycelium detecta o número de CPUs lógicas disponíveis e cria uma "shard" (partição) do índice para cada uma. Cada documento (neste caso, cada linha de um arquivo de texto) é atribuído a uma shard específica. Esse design permite que a indexação e a busca ocorram em paralelo, onde cada thread trabalha independentemente em sua própria partição de dados, minimizando a contenção e maximizando a vazão.

### 2. O Processo de Indexação

1. **Descoberta de Arquivos:** O motor varre o diretório de entrada em busca de todos os arquivos `.txt`.
    
2. **Processamento em Chunks:** Cada arquivo é lido em "chunks" (blocos de linhas) para otimizar o uso de memória.
    
3. Pipeline de Análise (Paralelo): Cada linha dentro de um chunk é processada por uma thread do pool do Rayon:
    
    a. A linha é tokenizada (dividida em palavras).
    
    b. Cada token é normalizado: convertido para minúsculas e tem a pontuação removida.
    
    c. Stop words são filtradas.
    
    d. O stemming é aplicado a cada token restante usando um Stemmer local da thread (thread_local!) para evitar contenção.
    
4. **Construção do Índice Invertido:** Para cada documento, o motor cria uma lista de termos e suas posições. Esses dados são então inseridos na shard apropriada. O dicionário de termos global (`TermDictionary`) é atualizado de forma segura usando um `Mutex`.
    
5. **Otimização Pós-Indexação:** Após o processamento de todos os arquivos:
    
    - As **frequências de documento** (`doc_frequencies`) para cada termo são pré-calculadas globalmente. Isso acelera drasticamente o cálculo do IDF durante a busca.
        
    - As listas de postings (ocorrências de termos) dentro de cada shard são **ordenadas por `doc_id`**, permitindo buscas binárias extremamente rápidas durante a fase de consulta.
        

### 3. O Algoritmo de Ranking

A relevância de um resultado é calculada combinando duas métricas:

- Pontuação BM25 (BM25): Um algoritmo de ranking de última geração que calcula a relevância com base na Frequência do Termo (TF) e na Frequência Inversa do Documento (IDF). Ele equilibra a frequência de um termo em um documento com sua raridade em toda a coleção de documentos. A fórmula do IDF usada é a variante BM25+, que é mais estável:
    
    $$ IDF(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right) $$
    
    Onde N é o número total de documentos e n(t) é o número de documentos que contêm o termo t.
    
- Boost de Proximidade: Mycelium vai além do BM25. Ele analisa as posições dos termos da consulta dentro dos documentos correspondentes. Se os termos da busca aparecem próximos uns dos outros, a pontuação final do documento é "impulsionada" (aumentada). Isso ajuda a priorizar resultados que são contextualmente mais relevantes. A pontuação final é calculada como:
    
    $$ \text{Pontuação Final} = \text{Pontuação BM25} \times (1 + \text{Boost de Proximidade}) $$
    

## 🚀 Instalação e Uso

### Pré-requisitos

- [Rust](https://www.rust-lang.org/tools/install) (versão 1.56 ou superior)
    

### Instalação

1. Clone este repositório:
    
    Bash
    
    ```
    git clone https://github.com/seu-usuario/mycelium.git
    cd mycelium
    ```
    
2. Construa o projeto em modo de release para máxima performance:
    
    Bash
    
    ```
    cargo build --release
    ```
    
    O binário estará disponível em `target/release/mycelium`.
    

### Como Usar

Mycelium possui dois subcomandos principais: `index` e `search`.

#### 1. Indexar um Diretório

Primeiro, crie um índice a partir de um diretório contendo arquivos `.txt`.

Bash

```
# Sintaxe:
# ./target/release/mycelium index <CAMINHO_DO_DIRETÓRIO> -o <ARQUIVO_DE_ÍNDICE_SAÍDA>

# Exemplo: Indexar uma pasta chamada 'meus_logs' e salvar como 'idx.bin'
./target/release/mycelium index ./meus_logs -o idx.bin
```

Você verá uma barra de progresso e, ao final, uma mensagem indicando que o índice foi salvo com sucesso.

```
Iniciando indexação com 12 shards (núcleos de CPU)...
[00:00:05] [████████████████████████████████████████] 50/50 (0s) Finalizando a indexação...
Processamento dos dados de indexação concluído em 5.12s.
Índice salvo em "idx.bin" (Tamanho: 15.23 MB) em 250ms.
```

#### 2. Buscar no Índice

Depois que o índice for criado, você pode usá-lo para fazer buscas.

Bash

```
# Sintaxe:
# ./target/release/mycelium search <CONSULTA> -i <ARQUIVO_DE_ÍNDICE>

# Exemplo: Buscar por "erro de conexão" no índice 'idx.bin'
./target/release/mycelium search "erro de conexão" -i idx.bin
```

Os 10 resultados mais relevantes serão impressos no console, classificados por pontuação.

```
Encontrados 128 resultados em 15.3ms:
  - Pontuação: 15.8741, Arquivo: "meus_logs/server.log", Linha: 8734
    > [2025-07-03] CRITICAL: erro fatal de conexão com o banco de dados principal.
  - Pontuação: 14.9823, Arquivo: "meus_logs/app.log", Linha: 102
    > [WARN] falha na conexão, tentando novamente... erro: timeout.
  - Pontuação: 12.1102, Arquivo: "meus_logs/server.log", Linha: 4501
    > [ERROR] erro de autenticação na conexão do cliente 192.168.1.10.
...
```

**Dica:** Para uma análise mais detalhada, use os flags de verbosidade:

- `-v` para logs de `INFO`.
    
- `-vv` para logs de `DEBUG`, que mostrarão os tokens da consulta, IDs e outras informações internas.
    

## 📦 Dependências Chave

Mycelium é construído sobre o incrível ecossistema de código aberto do Rust. As dependências mais importantes incluem:

- **[rayon](https://github.com/rayon-rs/rayon):** Para um paralelismo de dados fácil e poderoso.
    
- **[clap](https://github.com/clap-rs/clap):** Para construir a interface de linha de comando.
    
- **[serde](https://github.com/serde-rs/serde) & [bincode](https://github.com/bincode-org/bincode):** Para serialização e desserialização binária de alta performance.
    
- **[zstd](https://github.com/gyscos/zstd-rs):** Para compressão e descompressão rápida.
    
- **[ahash](https://github.com/tkaitchuck/aHash):** Para um hashing de `HashMap` extremamente rápido.
    
- **[indicatif](https://github.com/console-rs/indicatif):** Para belas barras de progresso no terminal.
    
- **[regex](https://github.com/rust-lang/regex), [rust-stemmers](https://github.com/CurrySoftware/rust-stemmers) & [once_cell](https://github.com/matklad/once_cell):** Para o pipeline de processamento de texto.
    
