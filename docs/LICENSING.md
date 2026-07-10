# Licensing — estado y restricciones

**No hay una licencia definida todavía.** No existe archivo `LICENSE` en la raíz a
propósito: la decisión requiere revisión humana porque el proyecto combina componentes con
restricciones propias que **no** son todas compatibles con una licencia permisiva única.

## Por qué no hay LICENSE aún

El repositorio integra o depende de varios elementos con términos de terceros:

| Componente | Restricción a verificar |
|---|---|
| **ViSpeR** (modelo base 288M y su código) | licencia del release de ViSpeR/TII — revisar si permite redistribución y uso comercial |
| **Modelo 50M (Gimeno) / LIP-RTVE** | términos de los pesos y del corpus base |
| **mpc001 / CMU-MOSEAS** | licencia del repo y del dataset multilingüe |
| **Dataset de YouTube** | los clips derivan de contenido de YouTube; sus **Términos de Servicio** y el copyright de cada autor limitan la redistribución. Por eso el dataset **no** se versiona (ver [`DATA_AND_ARTIFACTS.md`](DATA_AND_ARTIFACTS.md)) |
| **qwen3 (Ollama)** | licencia del modelo del corrector |
| **Grabaciones personales** | privadas, nunca se versionan |

## Implicancia práctica (hasta que haya decisión)

- **El código propio** de este repo (pipelines, demo, scripts) es del equipo autor, pero
  no se declara reutilizable bajo una licencia abierta hasta resolver lo de arriba.
- **No se redistribuyen** pesos de terceros ni el dataset dentro de Git.
- Quien quiera reutilizar debe: (1) obtener por su cuenta los pesos/datasets de sus fuentes
  originales bajo sus licencias, y (2) contactar a los autores para el uso del código.

## Qué falta decidir (humano)

1. Elegir una licencia para el **código propio** compatible con las dependencias
   (candidatas típicas: MIT/Apache-2.0 para el código si las dependencias lo permiten en
   tiempo de ejecución; o una licencia no comercial si algún componente lo exige).
2. Confirmar que esa elección no contradice la licencia de ViSpeR ni de los datasets.
3. Recién entonces agregar el archivo `LICENSE` en la raíz y el campo `license` en
   [`CITATION.cff`](../CITATION.cff).

Mientras tanto, este documento es la referencia de licenciamiento del proyecto.
