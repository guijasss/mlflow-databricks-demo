# Fluxo dos Pipelines

Este projeto usa GitHub Actions para validar e publicar o Databricks Asset
Bundle em dois ambientes no mesmo workspace Databricks:

* `qas`, usando o catalogo `qas_main`.
* `prd`, usando o catalogo `prd_main`.

Os targets ficam definidos em `databricks.yml`, e o workflow fica em
`.github/workflows/databricks-bundle.yml`.

## Branches

* `development`: branch de desenvolvimento compartilhada, quando usada.
* `feat/*`: branches de feature.
* `qas`: branch de homologacao.
* `master`: branch de producao.

O fluxo esperado e:

```text
development ou feat/* -> qas -> master
```

## Pull request para qas

Trigger:

```yaml
pull_request:
  branches:
    - qas
```

Condicao do job:

```text
base = qas
head = qualquer branch
```

Job executado:

* `validate-qas`

O que ele faz:

* faz checkout do codigo da PR;
* configura Python 3.13;
* roda checagem de sintaxe com `python -m compileall train.py predict.py src`;
* instala o Databricks CLI;
* roda `databricks bundle validate --target qas`.

Objetivo:

* garantir que o codigo de desenvolvimento pode ser promovido para QAS;
* validar o DAB usando as variaveis do target `qas`, incluindo catalogo
  `qas_main` e endpoint `fraud-detection-qas`.

Este trigger nao faz deploy.

## Tag qas-*

Trigger:

```yaml
push:
  tags:
    - qas-*
```

Job executado:

* `deploy-qas`

O que ele faz:

* faz checkout completo do repositorio, incluindo historico;
* verifica se o commit da tag pertence ao historico da branch `qas`;
* instala o Databricks CLI;
* roda `databricks bundle validate --target qas`;
* roda `databricks bundle deploy --target qas`.

Objetivo:

* publicar em QAS apenas commits que ja foram incorporados na branch `qas`;
* impedir deploy acidental de uma tag `qas-*` criada sobre um commit fora do
  fluxo de homologacao.

Exemplo:

```bash
git checkout qas
git pull origin qas
git tag qas-v0.1.0
git push origin qas-v0.1.0
```

## Pull request de qas para master

Trigger:

```yaml
pull_request:
  branches:
    - master
```

Condicao do job:

```text
base = master
head = qas
```

Job executado:

* `validate-prd`

O que ele faz:

* faz checkout do codigo da PR;
* configura Python 3.13;
* roda checagem de sintaxe com `python -m compileall train.py predict.py src`;
* instala o Databricks CLI;
* roda `databricks bundle validate --target prd`.

Objetivo:

* validar que o mesmo codigo aprovado em QAS tambem e valido para PRD;
* validar o DAB usando as variaveis do target `prd`, incluindo catalogo
  `prd_main` e endpoint `fraud-detection-prd`.

Este trigger nao faz deploy.

## Tag prd-* ou prod-*

Trigger:

```yaml
push:
  tags:
    - prd-*
    - prod-*
```

Job executado:

* `deploy-prd`

O que ele faz:

* faz checkout completo do repositorio, incluindo historico;
* verifica se o commit da tag pertence ao historico da branch `master`;
* instala o Databricks CLI;
* roda `databricks bundle validate --target prd`;
* roda `databricks bundle deploy --target prd`.

Objetivo:

* publicar em PRD apenas commits que ja foram incorporados na branch `master`;
* garantir que uma tag de producao nao implante codigo que ainda nao passou
  pelo merge formal para `master`.

Exemplo:

```bash
git checkout master
git pull origin master
git tag prd-v0.1.0 qas-v0.1.0
git push origin prd-v0.1.0
```

Criar a tag `prd-*` apontando para a tag `qas-*` garante que PRD implante o
mesmo commit aprovado em QAS.

## Execucao manual

Trigger:

```yaml
workflow_dispatch:
  inputs:
    target:
      options:
        - qas
        - prd
```

Jobs possiveis:

* `deploy-qas`, quando `target = qas`.
* `deploy-prd`, quando `target = prd`.

Objetivo:

* permitir reexecucao manual de deploy para um target especifico.

Como este trigger nao esta associado a uma tag, ele nao executa a verificacao
de pertencimento do commit a `qas` ou `master`. Use com cuidado e, se possivel,
proteja os GitHub environments `qas` e `prd` com aprovacao manual.

## Autenticacao

O workflow usa OIDC entre GitHub Actions e Databricks.

Permissoes do workflow:

```yaml
permissions:
  contents: read
  id-token: write
```

Variaveis esperadas no GitHub:

* `DATABRICKS_HOST`
* `DATABRICKS_CLIENT_ID`

Variaveis de ambiente usadas pelo workflow:

```yaml
DATABRICKS_AUTH_TYPE: github-oidc
DATABRICKS_HOST: ${{ vars.DATABRICKS_HOST }}
DATABRICKS_CLIENT_ID: ${{ vars.DATABRICKS_CLIENT_ID }}
```

O service principal configurado no Databricks precisa ter permissao para:

* validar e publicar bundles no workspace;
* criar ou atualizar os jobs do bundle;
* usar ou criar objetos nos catalogos `qas_main` e `prd_main`;
* criar os schemas `feature_store` e `models`;
* criar o volume `models.audit`;
* criar ou atualizar os endpoints `fraud-detection-qas` e
  `fraud-detection-prd`.

## Promocao recomendada

Fluxo completo:

```text
1. Abrir PR development -> qas ou feat/* -> qas.
2. Validar e aprovar a PR.
3. Fazer merge em qas.
4. Criar tag qas-* no commit de qas.
5. Aguardar deploy em QAS.
6. Validar funcionalmente QAS.
7. Abrir PR qas -> master.
8. Fazer merge em master sem squash/rebase.
9. Criar tag prd-* apontando para a tag qas-* aprovada.
10. Aguardar deploy em PRD.
```

O merge de `qas` para `master` deve preservar o commit aprovado em QAS. Se o
merge for feito com squash ou rebase, a tag `prd-*` apontando para a tag
`qas-*` pode nao pertencer ao historico de `master`, e o workflow de PRD vai
falhar na etapa de verificacao.
