# Guia Rápido do UV

O `uv` é um gerenciador de projetos e pacotes Python extremamente rápido. Aqui estão os comandos essenciais e mais utilizados.

## 1. Configuração e Sincronização

- **Criar ambiente e instalar dependências do `pyproject.toml`:**
  ```bash
  uv sync
  ```
  *(Cria a pasta `.venv` automaticamente caso não exista).*

- **Criar apenas uma venv vazia:**
  ```bash
  uv venv
  ```
- **Ativar a venv manualmente (Windows):**
  ```bash
  .venv\Scripts\activate
  ```

## 2. Gerenciar Dependências

- **Adicionar um novo pacote ao projeto:**
  ```bash
  uv add pandas
  ```
- **Adicionar um pacote apenas para desenvolvimento (ex: testes/formatação):**
  ```bash
  uv add --dev pytest ruff
  ```
- **Remover um pacote do projeto:**
  ```bash
  uv remove pandas
  ```

## 3. Executar Código

Você não precisa ativar a venv para rodar código se usar o `uv run`. Ele detecta o ambiente automaticamente.

- **Rodar um script Python:**
  ```bash
  uv run script.py
  ```
- **Rodar um módulo ou ferramenta (ex: Jupyter, Streamlit):**
  ```bash
  uv run streamlit run dashboard/streamlit_app.py
  ```

## 4. Atualizações

- **Atualizar as dependências e o `uv.lock`:**
  ```bash
  uv lock --upgrade
  ```