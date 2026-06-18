Foi baixado o .venv e o "pip install "fastapi[standard]" " no terminal. Seguindo o menual do proprio site FAST API

### comando do .venv

- python3 -m venv .venv
- source .venv/bin/activate

## Nota 
Quando você instala com pip install "fastapi[standard]" ele vem com algumas dependências padrão opcionais, incluindo fastapi-cloud-cli, que permite que você implante em Nuvem FastAPI.

Se você não quiser ter essas dependências opcionais, poderá instalar pip install fastapi.

Se você quiser instalar as dependências padrão, mas sem o fastapi-cloud-cli, você pode instalar com pip install "fastapi[standard-no-fastapi-cloud-cli]".

-----


Com o (.venv) ainda funionando, preciso intalr a IA que estarei usando, no caso a IA tucano 

- pip install transformers torch accelerate sentence-transformers
- pip install transformers torch accelerate fastapi uvicorn


-----

## executar no terminal quando todos os código funcionassem 

- fastapi dev
- fastapi dev main.py

