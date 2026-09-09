# -*- coding: utf-8 -*-
"""
Montagem dinâmica do prompt do AssistIA.

Em vez de um prompt_template gigante e estático com regras para TODOS os
interesses/sensibilidades/recursos possíveis, este módulo monta o prompt
em Python, injetando SOMENTE as regras relevantes para o que foi
selecionado no formulário para este aluno específico. Isso mantém o
prompt final enxuto (bom para modelos pequenos como o llama3.2) mesmo
que o catálogo de interesses/sensibilidades/recursos cresça no futuro.
"""

import unicodedata


def _normalizar(texto: str) -> str:
    """
    Normaliza uma string de opção (área, interesse, sensibilidade, recurso):
    remove acentos, espaços nas pontas e converte para minúsculas.

    Isso protege o sistema contra variações de digitação (ex: "Comunicação",
    "comunicacao ", "COMUNICAÇÃO") que, sem normalização, fariam a opção
    NÃO bater com as chaves dos dicionários (ex: "comunicacao") e ser
    silenciosamente ignorada do prompt — sem gerar nenhum erro visível.
    O formulário web já envia valores fixos sem acento, mas o modo CLI
    aceita texto livre, então essa normalização é a rede de segurança.
    """
    if not texto:
        return texto
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto


def _normalizar_lista(lista):
    return [_normalizar(item) for item in (lista or []) if item and item.strip()]


# 1. NÍVEIS DE SUPORTE (DSM-5)


NIVEIS_DSM5 = {
    "1": (
        "NÍVEL 1 (Requer suporte): comunicação com frases completas, mas com "
        "dificuldade em iniciar interações sociais. Adaptações: estímulo à "
        "iniciativa, mediação social, suporte para transições."
    ),
    "2": (
        "NÍVEL 2 (Requer suporte substancial): fala limitada a frases "
        "simples, dificuldade grave em comunicação não verbal e social. "
        "Adaptações: prancha de comunicação, figuras, rotina visual "
        "estruturada, professor como mediador constante."
    ),
    "3": (
        "NÍVEL 3 (Requer suporte muito substancial): comunicação muito "
        "limitada ou ausente, interações sociais raras. Adaptações: "
        "comunicação alternativa (PECS, prancha), ambiente previsível, "
        "apoio individualizado intenso."
    ),
}

# 2. INTERESSES -> materiais concretos (bate com os checkboxes do formulário)


INTERESSES_MATERIAIS = {
    "musica": {
        "usar": "instrumentos musicais reais adequados à idade (pandeiro, chocalho, tamborim, xilofone), potes com grãos bem fechados, colheres de madeira",
        "evitar": "frutas como instrumentos (ex: 'toque a maçã'), papelão como instrumento, sons muito altos ou estridentes",
    },
    "instrumentos": {
        "usar": "instrumentos musicais reais adequados à idade (pandeiro, chocalho, tamborim, xilofone, teclado infantil)",
        "evitar": "objetos que imitam instrumentos sem produzir som real, sons muito altos ou estridentes",
    },
    "desenhos": {
        "usar": "papel grande, giz de cera grosso, lápis de cor, canetinhas laváveis",
        "evitar": "materiais que não sejam de desenho, canetas permanentes",
    },
    "pintura": {
        "usar": "tintas atóxicas e laváveis, pincéis grandes, papel grosso, potes de água",
        "evitar": "tintas com cheiro forte, materiais que não sejam de pintura",
    },
    "massinha": {
        "usar": "massinha de modelar (3 cores diferentes), palitos de picolé, forminhas, rolos, cortadores de massinha",
        "evitar": "outros brinquedos no lugar da massinha",
    },
    "dinossauros": {
        "usar": "dinossauros de brinquedo ou papelão, figuras de dinossauros, pegadas desenhadas, ovos de dinossauro (papel)",
        "evitar": "carrinhos ou outros brinquedos não relacionados",
    },
    "animais": {
        "usar": "figuras de animais, brinquedos de animais, fotos de animais, fantoches de animais",
        "evitar": "animais de verdade sem supervisão, temas não relacionados a animais",
    },
    "natureza": {
        "usar": "folhas, galhos, flores, pedras, terra, areia, sementes, gravetos, plantas, conchas, algodão, água",
        "evitar": "potes e colheres como material principal, papelão como substituto do material natural",
    },
    "leitura": {
        "usar": "livros com imagens grandes e coloridas, histórias curtas (2-3 páginas), figuras da história",
        "evitar": "livros com texto longo, histórias complexas",
    },
    "jogos": {
        "usar": "jogos de encaixe, quebra-cabeças simples (2-4 peças), jogos de memória (pares), jogos de parear",
        "evitar": "jogos que exijam leitura, regras complexas ou mais de 2 jogadores",
    },
    "tecnologia": {
        "usar": "tablet ou computador com aplicativos educativos visuais, vídeos curtos, jogos educativos digitais simples",
        "evitar": "telas com estímulos rápidos/piscantes, tempo de tela acima de 5-10 minutos por atividade",
    },
    "esportes": {
        "usar": "bolas macias, bambolês, cones, fitas para marcar percurso",
        "evitar": "esportes com regras complexas ou que exijam grande número de jogadores",
    },
}

INTERESSE_FALLBACK = (
    "USE materiais concretos, específicos e realistas ligados a esse interesse "
    "(nunca genéricos como 'brinquedos'); NUNCA substitua por outro tema."
)

# 3. SENSIBILIDADES SENSORIAIS -> restrições (bate com os checkboxes)


SENSIBILIDADES_RESTRICOES = {
    "sons_altos": "EVITE instrumentos ou materiais que produzam sons altos/estridentes (ex: chocalhos fortes, apitos, brinquedos com alarme). Prefira sons suaves controlados pelo professor.",
    "luzes_fortes": "EVITE telas muito brilhantes, luzes piscantes ou ambientes com luz forte direta. Se usar tecnologia, reduza o brilho da tela.",
    "texturas": "EVITE materiais de textura desconfortável (ex: massinha muito pegajosa, tecidos ásperos, areia muito fina) sem antes considerar a tolerância do aluno.",
    "multidoes": "Realize a atividade em ambiente individual ou com poucos colegas, evitando grandes grupos.",
    "cheiros": "EVITE materiais perfumados ou com cheiro forte (ex: massinha aromatizada, canetinhas com cheiro, tintas com odor forte).",
    "toque": "EVITE contato físico direto não solicitado (ex: segurar a mão do aluno à força). Use objetos como intermediários e avise antes de tocar.",
    "movimento": "EVITE atividades com movimento corporal intenso ou giros. Prefira atividades sentadas e estáticas.",
    "comidas": "NUNCA use alimentos como material da atividade.",
    "rotina": "Reforce ainda mais o quadro de rotina visual e avise sobre qualquer mudança com antecedência, mesmo pequena.",
    "temperatura": "EVITE materiais muito frios ou muito quentes (ex: gelo, massinha recém-aquecida) sem checar tolerância antes.",
}

# 4. RECURSOS DISPONÍVEIS -> como podem ser usados (bate com os checkboxes)


RECURSOS_SUGESTOES = {
    "tablet": "pode usar aplicativos de comunicação alternativa (CAA), jogos educativos visuais e vídeos curtos como apoio.",
    "computador": "pode usar softwares educativos e apresentações visuais simples.",
    "impressora": "pode imprimir pranchas de comunicação, quadros de rotina e figuras personalizadas para este aluno.",
    "lousa_digital": "pode usar para mostrar sequências visuais grandes e interativas.",
    "reciclaveis": "priorize potes, caixas, rolos de papel e tampinhas como materiais complementares de baixo custo.",
    "livros": "pode incluir livros com imagens grandes como apoio visual complementar.",
    "jogos_educativos": "pode incluir jogos de encaixe, quebra-cabeça simples e jogos de parear.",
    "instrumentos_musicais": "pode incluir instrumentos reais como pandeiro, chocalho, tamborim.",
    "arte": "pode incluir giz de cera, tintas atóxicas, pincéis grandes.",
    "esporte": "pode incluir bolas macias, bambolês, cones para atividades motoras.",
}

# 5. ÁREAS DE ATENÇÃO -> foco de cada atividade (bate com o checkbox "areas")


AREAS_INSTRUCOES = {
    "comunicacao": (
        "Foco: comunicação expressiva/receptiva. Siga a REGRA DE COMUNICAÇÃO "
        "conforme o nível do aluno (ver seção específica abaixo)."
    ),
    "sensorial": (
        "Foco: regulação sensorial. Use estímulos controlados (texturas, sons, "
        "movimento) SEMPRE respeitando as SENSIBILIDADES SENSORIAIS informadas — "
        "nunca proponha um estímulo que esteja na lista de restrições."
    ),
    "motor": (
        "Foco: coordenação motora fina ou global. Use ações concretas como "
        "pegar, encaixar, amassar, apertar, empilhar ou lançar — adequadas à "
        "idade, sem objetos cortantes se o aluno tiver até 5 anos."
    ),
    "cognitivo": (
        "Foco: atenção, memória ou sequenciamento. Prefira atividades curtas "
        "(3 a 5 minutos), com pausas e alternância — jogos de esconder/achar, "
        "pareamento ou sequência simples."
    ),
    "social": (
        "Foco: interação social / redução de isolamento. O passo [PROFESSOR] "
        "DEVE conter literalmente a frase 'senta-se ao lado do aluno' (ou "
        "'posiciona-se ao lado do aluno') — não é opcional, é uma frase "
        "obrigatória no texto do passo. A atividade é feita JUNTOS do início "
        "ao fim, usando o interesse do aluno como ponte — o aluno NUNCA "
        "realiza a etapa central sozinho."
    ),
    "estrutura": (
        "Foco: rotina e estrutura. Crie um quadro de rotina com FIGURAS/"
        "pictogramas (nunca texto) representando atividades REAIS do dia "
        "escolar (entrada, lanche, atividade, saída), mostrado ANTES de cada "
        "transição."
    ),
    "alfabetizacao": (
        "Foco: reconhecimento de letras, associação som-letra. Use letras "
        "grandes, cartões, associação com figuras. EVITE escrita para alunos "
        "não verbais."
    ),
    "matematica": (
        "Foco: contagem, pareamento numérico, quantidades. Use objetos "
        "concretos para contar, números grandes. EVITE operações abstratas "
        "sem suporte visual."
    ),
    "logica": (
        "Foco: sequenciamento, causa-efeito, categorização. Use quebra-cabeças, "
        "jogos de encaixe, sequência de imagens. EVITE regras complexas ou "
        "muitos passos."
    ),
    "emocional": (
        "Foco: reconhecimento de emoções, autorregulação. Use figuras de "
        "emoções, espelho, respiração guiada. EVITE confronto ou pressão "
        "para falar sobre sentimentos."
    ),
    "regulacao": (
        "Foco: acalmar, organizar o corpo, reduzir agitação. Use atividades "
        "repetitivas, estímulos suaves, pausas. EVITE estímulos intensos ou "
        "atividades competitivas."
    ),
}

AREAS_CONTEXTO_APLICACAO = {
    "comunicacao": "no início da aula, antes de iniciar outras atividades",
    "sensorial": "durante momentos de agitação ou transição, como uma pausa sensorial",
    "motor": "no momento de brincar dirigido, entre outras atividades",
    "cognitivo": "em intervalos curtos entre uma tarefa e outra",
    "social": "durante o intervalo ou momento de brincar livre, sempre com o professor ao lado",
    "estrutura": "sempre antes de cada transição de atividade, ao mostrar o quadro de rotina",
    "alfabetizacao": "durante o momento de leitura ou escrita dirigida",
    "matematica": "durante atividades de contagem ou números",
    "logica": "em momentos de jogos ou desafios",
    "emocional": "após momentos de estresse ou frustração",
    "regulacao": "durante momentos de agitação ou antes de atividades intensas",
}

# checkbox "areas" — este mapa reconcilia os dois.
AREA_PRINCIPAL_PARA_AREA = {
    "comunicacao": "comunicacao",
    "regulacao_sensorial": "sensorial",
    "motor": "motor",
    "cognitivo": "cognitivo",
    "interacao_social": "social",
    "estruturacao": "estrutura",
}

PRIORIDADE_FREQUENCIA = {
    "alta": "diariamente",
    "media": "2 a 3 vezes por semana",
    "baixa": "1 vez por semana, ou quando a situação surgir",
}


# 6. FUNÇÕES QUE MONTAM CADA SEÇÃO DINÂMICA DO PROMPT


def montar_secao_interesses(interesses_list, interesses_observacao=""):
    linhas = []
    for interesse in interesses_list:
        dados = INTERESSES_MATERIAIS.get(interesse)
        if dados:
            linhas.append(
                f"- {interesse.upper()}: USE {dados['usar']}. "
                f"NUNCA use {dados['evitar']}."
            )
        else:
            linhas.append(f"- {interesse.upper()}: {INTERESSE_FALLBACK}")

    if interesses_observacao:
        linhas.append(
            f'- OUTRO INTERESSE INFORMADO PELO PROFESSOR: "{interesses_observacao}" '
            f"— {INTERESSE_FALLBACK}"
        )

    if not linhas:
        return "Nenhum interesse específico informado — baseie-se no PERFIL DO ALUNO."

    return "\n".join(linhas)


def montar_secao_sensibilidades(sensibilidades_list, sensibilidades_observacao=""):
    linhas = [
        f"- {SENSIBILIDADES_RESTRICOES[s]}"
        for s in sensibilidades_list
        if s in SENSIBILIDADES_RESTRICOES
    ]
    if sensibilidades_observacao:
        linhas.append(
            f'- OUTRA SENSIBILIDADE INFORMADA: "{sensibilidades_observacao}" — '
            f"evite qualquer material ou situação que possa causar desconforto "
            f"relacionado a isso."
        )
    if not linhas:
        return "Nenhuma sensibilidade sensorial informada — sem restrições adicionais."
    return "\n".join(linhas)


def montar_secao_recursos(recursos_list, recursos_observacao=""):
    linhas = [
        f"- {r.upper()}: {RECURSOS_SUGESTOES[r]}"
        for r in recursos_list
        if r in RECURSOS_SUGESTOES
    ]
    if recursos_observacao:
        linhas.append(f'- OUTRO RECURSO INFORMADO: "{recursos_observacao}"')
    if not linhas:
        return "Nenhuma tecnologia ou recurso extra disponível — use apenas materiais físicos simples e seguros."
    return "\n".join(linhas)


def montar_secao_necessidades(areas_list, area_principal=None):
    area_principal_normalizada = AREA_PRINCIPAL_PARA_AREA.get(area_principal)

    linhas = []
    for area in areas_list:
        instrucao = AREAS_INSTRUCOES.get(area)
        if not instrucao:
            continue
        marcador = " ⭐ (ÁREA PRIORITÁRIA — dê mais destaque e detalhe a esta atividade)" if area == area_principal_normalizada else ""
        contexto = AREAS_CONTEXTO_APLICACAO.get(area)
        linha_contexto = f" | USE ESTE CONTEXTO NO COMO APLICAR: {contexto}" if contexto else ""
        linhas.append(f"- {area.upper()}{marcador}: {instrucao}{linha_contexto}")

    if not linhas:
        return "Nenhuma área de atenção marcada — identifique as necessidades a partir das DIFICULDADES e da PERGUNTA DO PROFESSOR."

    return "\n".join(linhas)


# 7. FUNÇÃO DE DICAS ESPECÍFICAS POR ÁREA

def get_dicas_especificas_por_area(area, indice_atividade, total_atividades):
    """Retorna dicas específicas para cada área para evitar repetição"""

    # Dicas base por área
    dicas_base = {
        "comunicacao": {
            "foco": "escolha dirigida, apontar figuras, comunicação não verbal",
            "use": "prancha de comunicação, figuras, objetos concretos",
            "evite": "atividades que exijam fala ou perguntas abertas",
            "exemplo": "aluno aponta para a figura do dinossauro que quer usar"
        },
        "social": {
            "foco": "interação com o professor, atividades JUNTOS",
            "use": "interesse do aluno como ponte, revezamento",
            "evite": "aluno fazendo atividade sozinho",
            "exemplo": "professor e aluno montam juntos o quebra-cabeça"
        },
        "motor": {
            "foco": "coordenação fina (pegar, amassar, encaixar)",
            "use": "massinha, pinças, jogos de encaixe, recorte (se >5 anos)",
            "evite": "atividades muito longas ou cansativas",
            "exemplo": "aluno amassa a massinha para formar letras"
        },
        "cognitivo": {
            "foco": "memória, atenção, sequenciamento",
            "use": "jogos de memória, pareamento, sequência de figuras",
            "evite": "atividades que exijam leitura ou escrita",
            "exemplo": "aluno encontra o par da figura do dinossauro"
        },
        "sensorial": {
            "foco": "regulação, exploração tátil/auditiva/visual",
            "use": "texturas controladas, sons suaves, luzes suaves",
            "evite": "estímulos que estão na lista de restrições",
            "exemplo": "aluno toca folhas secas e macias"
        },
        "estrutura": {
            "foco": "rotina, previsibilidade, transições",
            "use": "quadro de rotina com FIGURAS (não texto)",
            "evite": "forçar o interesse do aluno aqui - use figuras do dia escolar",
            "exemplo": "mostrar figuras do dia: entrada, lanche, atividade"
        },
        "alfabetizacao": {
            "foco": "reconhecimento de letras, associação som-letra",
            "use": "letras grandes, cartões, associação com figuras",
            "evite": "escrita para alunos não verbais",
            "exemplo": "aluno aponta para a letra D de dinossauro"
        },
        "matematica": {
            "foco": "contagem, pareamento numérico, quantidades",
            "use": "objetos concretos para contar, números grandes",
            "evite": "operações abstratas sem suporte visual",
            "exemplo": "aluno conta quantos dinossauros tem na mesa"
        },
        "logica": {
            "foco": "sequenciamento, causa-efeito, categorização",
            "use": "quebra-cabeças, jogos de encaixe, sequência de imagens",
            "evite": "regras complexas ou muitos passos",
            "exemplo": "aluno coloca as figuras na ordem correta"
        },
        "emocional": {
            "foco": "reconhecimento de emoções, autorregulação",
            "use": "figuras de emoções, espelho, respiração guiada",
            "evite": "confronto ou pressão para falar sobre sentimentos",
            "exemplo": "aluno aponta para a figura que mostra como se sente"
        },
        "regulacao": {
            "foco": "acalmar, organizar o corpo, reduzir agitação",
            "use": "atividades repetitivas, estímulos suaves, pausas",
            "evite": "estímulos intensos ou atividades competitivas",
            "exemplo": "aluno faz movimentos lentos com as mãos"
        }
    }

    dica = dicas_base.get(area, {
        "foco": "objetivo específico da área",
        "use": "materiais apropriados",
        "evite": "o que não for adequado",
        "exemplo": "ação concreta"
    })

    # Instrução de diferenciação baseada no índice
    if total_atividades > 1:
        instrucao_diferenciacao = f"""
⚠️ IMPORTANTE: Esta é a atividade {indice_atividade + 1} de {total_atividades} atividades.

Para EVITAR atividades repetidas ou muito parecidas com as outras:
1. Use um NOME ÚNICO que NÃO se repita nas outras atividades
2. Use MATERIAIS DIFERENTES das outras atividades
3. Use uma ABORDAGEM DIFERENTE (ex: se a outra usou massinha, esta use desenho)
4. O COMO FAZER deve ter PASSOS DIFERENTES

DICAS ESPECÍFICAS PARA ESTA ÁREA ({area.upper()}):
- FOQUE em: {dica['foco']}
- USE: {dica['use']}
- EVITE: {dica['evite']}
- EXEMPLO DE AÇÃO: {dica['exemplo']}
"""
    else:
        instrucao_diferenciacao = f"""
DICAS ESPECÍFICAS PARA ESTA ÁREA ({area.upper()}):
- FOQUE em: {dica['foco']}
- USE: {dica['use']}
- EVITE: {dica['evite']}
- EXEMPLO DE AÇÃO: {dica['exemplo']}
"""

    return instrucao_diferenciacao


# 7B. GERAÇÃO POR ÁREA (uma chamada ao modelo por atividade)


'''Em vez de pedir 2-3 atividades numa única resposta (o que sobrecarrega
modelos pequenos rodando em CPU, causando contagem errada, área trocada,
 COMO APLICAR repetido etc.), esta função monta um prompt bem menor,
focado em UMA ÚNICA área/atividade por vez. O código que chama o modelo
(main.py) deve rodar isso em loop, uma vez por área selecionada, e juntar
os textos retornados'''


def escolher_interesse_para_area(interesses_list, indice_area):
    """
    Escolhe qual interesse usar nesta atividade específica, distribuindo
    por rodízio quando há mais de um interesse informado (ex: 2 interesses
    e 3 áreas -> área 0 usa interesse 0, área 1 usa interesse 1, área 2
    usa interesse 0 de novo). Retorna None se não houver interesse algum.
    """
    interesses_norm = _normalizar_lista(interesses_list)
    if not interesses_norm:
        return None
    return interesses_norm[indice_area % len(interesses_norm)]


def montar_prompt_atividade_unica(
        descricao_aluno,
        nivel_dsm5,
        pergunta,
        base_conhecimento,
        area,
        eh_area_principal=False,
        prioridade="media",
        interesse=None,
        sensibilidades_list=None,
        sensibilidades_observacao="",
        recursos_list=None,
        recursos_observacao="",
        indice_atividade=0,
        total_atividades=1,
        atividades_anteriores=None,
):
    area = _normalizar(area)
    interesse = _normalizar(interesse) if interesse else None
    sensibilidades_list = _normalizar_lista(sensibilidades_list)
    recursos_list = _normalizar_lista(recursos_list)
    nivel_dsm5 = (nivel_dsm5 or "").strip()

    if atividades_anteriores is None:
        atividades_anteriores = []

    nivel_suporte_detalhe = NIVEIS_DSM5.get(nivel_dsm5, "Nivel 1")
    frequencia = PRIORIDADE_FREQUENCIA.get(prioridade, "2 a 3 vezes por semana")
    contexto = AREAS_CONTEXTO_APLICACAO.get(area, "momento adequado")

    # Restricoes
    restricoes = []
    for s in sensibilidades_list:
        if s in SENSIBILIDADES_RESTRICOES:
            restricoes.append(SENSIBILIDADES_RESTRICOES[s])
    restricoes_texto = "; ".join(restricoes) if restricoes else "Nenhuma"

    recursos_texto = ", ".join([r.upper() for r in recursos_list]) if recursos_list else "Nenhum recurso especifico"

    # --- IDENTIFICAR MATERIAIS JA USADOS ---
    materiais_usados = []
    if atividades_anteriores:
        for ativ in atividades_anteriores:
            if "MATERIAIS:" in ativ:
                try:
                    parte = ativ.split("MATERIAIS:")[1].split("COMO FAZER:")[0].strip()
                    itens = [i.strip() for i in parte.split(",")[:3]]
                    materiais_usados.extend(itens)
                except:
                    pass

    # --- LISTA DE MATERIAIS DINAMICA BASEADA NO INTERESSE ---
    if interesse and interesse in INTERESSES_MATERIAIS:
        materiais_sugeridos = INTERESSES_MATERIAIS[interesse]['usar']
        todos_materiais = [item.strip() for item in materiais_sugeridos.split(',')]
    else:
        # Fallback generico - usar materiais do interesse se disponivel
        if interesse:
            todos_materiais = [f"material de {interesse} 1", f"material de {interesse} 2", f"material de {interesse} 3"]
        else:
            todos_materiais = ["material 1", "material 2", "material 3", "material 4", "material 5"]

    # Remover materiais ja usados
    materiais_disponiveis = []
    for m in todos_materiais:
        if m not in materiais_usados:
            materiais_disponiveis.append(m)

    # Pegar os primeiros 3 materiais disponiveis
    if len(materiais_disponiveis) >= 3:
        materiais_escolhidos = materiais_disponiveis[:3]
    else:
        materiais_escolhidos = todos_materiais[:3]

    materiais_str = ", ".join(materiais_escolhidos)

    # --- REGRA SOCIAL ---
    regra_social = ""
    if area == "social":
        regra_social = "Passo 1 do PROFESSOR: comeca com 'senta-se ao lado do aluno'."

    # --- REGRA PRANCHA ---
    regra_prancha = ""
    if nivel_dsm5 in ["2", "3"]:
        regra_prancha = "Inclua uma prancha de comunicacao com figuras."

    # --- DICA POR AREA ---
    dica_area = ""
    if area == "sensorial":
        dica_area = "Foco: exploracao de texturas. Aluno toca os materiais."
    elif area == "regulacao":
        dica_area = "Foco: acalmar e organizar. Aluno faz acoes lentas e repetitivas."
    elif area == "comunicacao":
        dica_area = "Foco: escolha dirigida. Aluno aponta para figuras ou objetos."
    elif area == "social":
        dica_area = "Foco: interacao. Atividade feita JUNTOS do inicio ao fim."
    elif area == "motor":
        dica_area = "Foco: coordenacao fina. Aluno pega, encaixa, amassa."
    elif area == "cognitivo":
        dica_area = "Foco: memoria e atencao. Aluno pareia, lembra, organiza."
    elif area == "alfabetizacao":
        dica_area = "Foco: reconhecimento de letras. Aluno aponta e pareia."
    elif area == "matematica":
        dica_area = "Foco: contagem e numeros. Aluno conta e quantifica."
    elif area == "logica":
        dica_area = "Foco: sequenciamento. Aluno coloca em ordem."
    elif area == "estrutura":
        dica_area = "Foco: rotina. Usa figuras do dia escolar."
    elif area == "emocional":
        dica_area = "Foco: reconhecimento de emocoes. Aluno aponta figuras."
    else:
        dica_area = f"Foco: objetivo especifico da area {area}."

    # --- CONSTRUIR AVISO DE MATERIAIS DIFERENTES ---
    aviso_materiais = ""
    if materiais_usados:
        aviso_materiais = f"""
MATERIAIS JA USADOS EM OUTRAS ATIVIDADES: {', '.join(materiais_usados)}
VOCE DEVE USAR MATERIAIS DIFERENTES.
MATERIAIS PARA ESTA ATIVIDADE: {materiais_str}
"""

    # --- PROMPT FINAL ---
    prompt = f"""CRIE UMA ATIVIDADE EDUCACIONAL PARA ALUNO COM TEA.

DADOS DO ALUNO:
{descricao_aluno}
NIVEL: {nivel_suporte_detalhe}
AREA: {area.upper()}
INTERESSE: {interesse.upper() if interesse else "Nenhum"}
PERGUNTA: {pergunta}

REGRA 1 - MATERIAIS DIFERENTES:
{aviso_materiais if aviso_materiais else "Use materiais relacionados ao interesse do aluno."}

REGRA 2 - ACOES CONCRETAS:
Use apenas: tocar, pegar, apontar, encaixar, mostrar, segurar, contar, parear.
NAO use: explorar, interagir, discutir, observar (sem acao).

REGRA 3 - {regra_prancha if regra_prancha else "Nenhuma regra adicional de comunicacao."}
REGRA 4 - {regra_social if regra_social else "Nenhuma regra social especifica."}
REGRA 5 - {dica_area}

FORMATO EXATO (sem markdown, sem negrito):

NOME DA ATIVIDADE: [nome unico]
OBJETIVO: [descricao especifica]
MATERIAIS: {materiais_str}
COMO FAZER:
  1. [PROFESSOR] [acao concreta]
  2. [ALUNO] [acao concreta]
  3. [JUNTOS] [acao concreta]
QUEM FAZ: [PROFESSOR] / [ALUNO] / [JUNTOS]
COMO APLICAR: {frequencia}, {contexto}

CRIE A ATIVIDADE:"""

    return prompt.strip()

# 8. EXEMPLO DE USO (para você testar isoladamente antes de integrar no main.py)

if __name__ == "__main__":
    exemplo = montar_prompt(
        descricao_aluno=(
            "Idade: 4 anos\n"
            "Comunicação: não verbal\n"
            "Interação social: não interage\n"
            "Dificuldades: isolamento\n"
            "Potencialidades: criatividade\n"
            "Observações: gosta de rotina"
        ),
        nivel_dsm5="2",
        pergunta="Como posso ajudar esse aluno com o seu isolamento, comunicação e a parte de manter a rotina?",
        base_conhecimento="Nenhum documento específico encontrado. Use conhecimento geral sobre TEA.",
        areas_list=["comunicacao", "social", "estrutura"],
        area_principal="interacao_social",
        prioridade="alta",
        interesses_list=["massinha"],
        sensibilidades_list=["sons_altos"],
        recursos_list=["reciclaveis"],
    )
    print("=== PROMPT ÚNICO (multi-atividade, versão antiga) ===")
    print(f"[tamanho: {len(exemplo)} caracteres]\n")

    print("\n=== PROMPTS POR ÁREA (versão nova, uma chamada por atividade) ===\n")
    areas_teste = ["comunicacao", "social", "estrutura"]
    atividades_anteriores = []
    for i, area in enumerate(areas_teste):
        interesse = escolher_interesse_para_area(["massinha"], i)
        p = montar_prompt_atividade_unica(
            descricao_aluno="Idade: 4 anos\nComunicação: não verbal\nDificuldades: isolamento",
            nivel_dsm5="2",
            pergunta="Como posso ajudar esse aluno com isolamento, comunicação e rotina?",
            base_conhecimento="Nenhum documento específico encontrado.",
            area=area,
            eh_area_principal=(area == "social"),
            prioridade="alta",
            interesse=interesse,
            sensibilidades_list=["sons_altos"],
            recursos_list=["reciclaveis"],
            indice_atividade=i,
            total_atividades=len(areas_teste),
            atividades_anteriores=atividades_anteriores
        )
        print(f"--- Prompt para área '{area}' ({len(p)} caracteres) ---")
        print(p[:500] + "...\n")
        # Simular que a atividade foi gerada para o próximo loop
        atividades_anteriores.append(f"NOME DA ATIVIDADE: Exemplo {area.upper()}")