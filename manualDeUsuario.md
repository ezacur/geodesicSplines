# Editor de Splines Geodésicas — Manual de Usuario

Este manual es la referencia práctica del editor interactivo.  Recorre
todas las funcionalidades que puedes manejar desde el ratón, el teclado
y la línea de comandos.  Si buscas algoritmos, notas de rendimiento o
cómo está organizado el código, consulta
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — ese documento está
orientado a desarrolladores.

---

## 1. ¿Para qué sirve?

El editor te permite dibujar curvas suaves **sobre una malla 3-D**
(un fichero OBJ / PLY / STL / VTK).  A diferencia de las herramientas
de curvas de Blender, Maya o la mayoría de los paquetes CAD, cada
punto de la curva está realmente sobre la superficie — no flota en
las concavidades, no atraviesa los pliegues, y su longitud de arco no
es una estimación que se desvía unos milímetros.  Esto importa cuando
luego necesitas la curva para:

- Cortar patrones de tela siguiendo la curva sobre un escaneo 3-D.
- Planificar la trayectoria de una fresadora CNC sobre un modelo
  esculpido.
- Ingeniería inversa de geometría medida (escaneos anatómicos, piezas
  CAD escaneadas, artefactos arqueológicos).
- Laminado de fibra de carbono a lo largo de una geodésica precisa.
- Cualquier flujo en el que "la distancia a lo largo de la curva"
  tenga que ser exacta.

Cada curva que colocas es **una cadena de nodos**, con **tiradores de
tangente** que te dan control C1-continuo sobre su forma (la misma
idea que en una Bézier 2-D, pero viviendo sobre la superficie).
Puedes tener varias splines independientes en la misma sesión,
guardarlas y recargarlas como JSON, y exportar la curva a CSV / OBJ /
VTK para herramientas externas.

---

## 2. Instalación

Instala las dependencias una vez.  Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
# Opcional, recomendado (acelera entre 50× y 2000× los caminos
# críticos):
pip install -e .[jit]
```

La dependencia más delicada es `potpourri3d`: necesita un compilador
C++17 en su primera instalación (MSVC 2022 Build Tools en Windows,
gcc ≥ 9 en Linux).  Para la mayoría de combinaciones plataforma /
Python existen *wheels* precompilados que pip selecciona automáticamente.

---

## 3. Lanzar el editor

El editor se invoca desde una terminal.  Cuatro formas equivalentes
cubren cualquier punto de partida:

```bash
# 1. Sin argumentos — abre fandisk.obj si está en el directorio
#    actual; si no, recurre a un icosaedro de demostración interno.
python geo_splines.py

# 2. Abrir una malla específica, sesión en blanco.
python geo_splines.py malla.ply

# 3. Reanudar una sesión guardada.  La ruta de la malla se lee del
#    campo "mesh_file" del JSON, así que el editor la encuentra
#    automáticamente.
python geo_splines.py sesion.json

# 4. Reanudar una sesión contra una malla distinta (útil para
#    inspeccionar las mismas splines sobre una superficie remallada
#    o de mayor resolución).
python geo_splines.py sesion.json malla_hires.vtk
```

Formatos de malla aceptados: `.obj`, `.ply`, `.stl`, `.vtk` (y
cualquier otro que el lector de PyVista reconozca).

El editor imprime su cabecera de versiones, el fichero recién cargado
y un **bloque de ayuda por consola** con todos los atajos — práctico
para repasar mientras encuentras las teclas.  Los mismos atajos viven
en un pequeño panel en pantalla que permanece visible mientras
trabajas.

---

## 4. La interfaz de un vistazo

Al abrirse la ventana ves:

- **La malla**, sombreada.
- Un **cursor de superficie** en la posición del ratón: un pequeño
  círculo sobre la malla con una cruz alineada con los ejes locales —
  es el punto 3-D sobre el que actuarías (doble clic para añadir un
  nodo, etc.).
- Una **línea de HUD** arriba a la izquierda en texto coloreado
  (mensajes de estado: "READY", "NODE INSERTED", "REFINED (EXACT)",
  "GUIDES LOADED ...", errores en rojo).
- El **panel de ayuda** arriba a la derecha (columna estrecha con la
  misma lista de atajos que se imprimió en consola).
- Tres **checkboxes** abajo a la izquierda para las capas de curva
  (azul / naranja / interp) — son espejo de las teclas `b` / `o` / `k`.

**Controles de cámara** estándar de VTK:

- **Botón izquierdo arrastrado** sobre el fondo: rotar alrededor del
  punto focal.
- **Botón central arrastrado** o **Shift+izquierdo**: paneo.
- **Rueda del ratón**: zoom.
- **Botón derecho arrastrado**: zoom (o roll, según la build de VTK).
- **R** en el teclado principal *no* está re-bindeado: el "reset
  camera" de PyVista está deshabilitado porque el editor usa `R` para
  reconstruir la capa naranja (ver §10).

---

## 5. Tu primera spline en 5 pasos

1. **Coloca el primer nodo**: doble clic izquierdo en cualquier
   punto de la superficie.  Aparece una esfera roja (el **nodo P**)
   con dos flechas coloreadas (los **tiradores de tangente A y B**,
   verde y azul respectivamente).  El HUD dice "NODE INSERTED".
2. **Coloca el segundo nodo**: doble clic izquierdo en otro punto.
   Aparece un segundo nodo y una curva suave entre los dos — en
   realidad **tres versiones** superpuestas:
     - **Azul** (visible por defecto): la curva interactiva, instantánea.
     - **Naranja** (calculada en segundo plano, puede tardar un segundo
       en aparecer): la versión fully-geodésica.  Es la curva "final".
     - **Negra** (se activa con `k`): una interpolación B-spline
       proyectada sobre la superficie.
3. **Ajusta el nodo**: arrastra la esfera roja (P).  La curva y sus
   tangentes siguen en tiempo real.  Al soltar el ratón se dispara un
   "refinado" de ~150 ms que reemplaza la previsualización en vivo
   por la geodésica exacta — el HUD parpadea "REFINED (EXACT)".
4. **Modela la curva**: arrastra uno de los tiradores (A o B).  El
   otro tirador se mantiene simétrico (la continuidad C1 se respeta
   automáticamente) y la curva se reforma.
5. **Guarda**: pulsa `s`.  Se escribe un fichero JSON con marca de
   tiempo en el directorio actual; el HUD reporta el nombre.
   Recárgalo cuando quieras con `l` (diálogo de fichero) o pasando la
   ruta por línea de comandos.

Ese es todo el bucle nuclear.  Todo lo que sigue lo extiende o lo
afina.

---

## 6. Trabajar con nodos

### Añadir nodos

- **Doble clic izquierdo sobre la superficie**: añade un nodo al final
  de la spline activa.
- **Doble clic izquierdo sobre el marcador de hover de una curva**:
  inserta un nodo **en la curva** en el punto exacto bajo el cursor.
  El marcador de hover es un pequeño **visor telescópico** que aparece
  cuando tu cursor está cerca de cualquier curva visible — ver §11.

### Eliminar nodos

- **Retroceso (Backspace)**: elimina el último nodo de la spline
  activa.
- Si la spline activa ya está vacía (p. ej. después de pulsar doble
  clic derecho para iniciar una nueva), Retroceso deshace ese "break"
  y te devuelve a la spline anterior.

### Edición de coordenadas exactas

- **Doble clic derecho sobre una esfera P**: abre un pequeño diálogo
  donde puedes teclear los `x`, `y`, `z` exactos que quieres.  Una
  previsualización muestra dónde proyecta el punto introducido sobre
  la superficie; pulsa OK para confirmar.

### Deshacer / Rehacer

- **Ctrl+Z**: deshace la última acción.  Se mantienen hasta 50 niveles
  de historial **para cualquier** mutación de spline — añadir /
  eliminar nodos, soltar arrastre, alternar capas, cargar sesión,
  editar coordenadas.
- **Ctrl+Y**: rehacer.

El undo usa diff de snapshots, así que incluso splines grandes
(cientos de nodos) se deshacen en pocos milisegundos.

---

## 7. Trabajar con los tiradores de tangente (A y B)

Cada nodo lleva dos flechas:

- **A** apunta "hacia atrás" en la curva (hacia el nodo anterior).
- **B** apunta "hacia delante" (hacia el siguiente).

Las flechas viven sobre la superficie, y al arrastrarlas se rota la
dirección de la tangente *manteniendo el tirador opuesto simétrico* —
así que la curva sigue siendo C1-continua (sin doblez en el nodo) sin
que tengas que hacer nada.

### Arrastre estándar del tirador

Coge una flecha y arrástrala.  La flecha bajo el cursor se vuelve
**negra** y crece ligeramente; todo el gizmo del nodo se ilumina a
opacidad total **y se eleva en el z-buffer** para dibujarse por
encima de cualquier curva naranja / azul / negra que pudiera
solaparlo — así lees la geometría de un vistazo sin que una curva
superpuesta tape los tiradores.

### Arrastre solo de magnitud (Shift)

A veces quieres cambiar *cuán larga* es la tangente (controla cuán
"agudo" es el doblez en el nodo) sin cambiar su **dirección**.  Mantén
**Shift** mientras arrastras A o B: la distancia del cursor al origen
del nodo se convierte en la nueva longitud de la tangente, pero la
dirección se preserva.  Cruza el origen y la tangente se voltea para
que el tirador siga visualmente a tu cursor.

> Los modificadores de snap a vértice / arista (descritos en la
> siguiente sección) solo aplican al nodo P, no a A / B.  Hacer snap
> de la longitud de la tangente a un vértice discreto rompería la
> sensación de scrub suave.

---

## 8. Modificadores de snap — aterrizar en rasgos exactos de la malla

Estos modificadores ayudan cuando necesitas que un nodo coincida con
una referencia precisa sobre la malla.

### Shift + arrastrar P → snap al vértice más cercano

Mantén **Shift** mientras arrastras una esfera P.  A medida que te
mueves, aparece una **esfera dorada** en el vértice más cercano de la
malla.  Suelta el ratón sobre la esfera dorada y el nodo aterriza
exactamente sobre ese vértice (origen = posición del vértice).  El
HUD muestra `SNAP → vertex <idx>`.

### Ctrl + arrastrar P → snap a la arista más cercana

Mantén **Ctrl** mientras arrastras.  Aparece una **esfera cian** sobre
la arista más cercana, en el pie perpendicular desde tu cursor
(restringido a los extremos de la arista).  Al soltar, el nodo
aterriza exactamente sobre ese punto de la arista.  El HUD muestra
`SNAP → edge <va>-<vb> t=<0.0–1.0>`.

Las aristas son aristas reales de la malla, así que el nodo permanece
exactamente sobre la superficie por construcción.  Cualquiera de los
dos modificadores desactiva el debouncing de la previsualización en
vivo — cada movimiento es exacto.

---

## 9. Múltiples splines y bucles cerrados

### Empezar una nueva spline

**Doble clic derecho sobre la superficie vacía** (no sobre un nodo).
Aparece un "break" vacío al final de la lista de splines y pasa a ser
la nueva spline activa.  Añade nodos como de costumbre.  La spline
anterior sigue visible pero sus tiradores se atenúan ligeramente para
comunicar "inactiva".

### Cambiar entre splines

Doble clic izquierdo sobre cualquier nodo de otra spline para hacer
esa spline activa.  Todas las demás se atenúan; solo la activa
muestra los tiradores de tangente a todo color.

### Cerrar una spline (hacer un bucle)

Cuando una spline tiene **3 o más nodos**, pulsa **C**.  El tirador A
del primer nodo se reutiliza como tangente de cierre hacia el último
nodo, se dibuja el tramo de cierre, y se auto-crea una nueva spline
vacía para que puedas empezar inmediatamente una nueva forma sin un
paso extra.  El HUD muestra `LOOP CLOSED + BREAK`.

Vuelve a pulsar **C** sobre una spline cerrada para reabrirla: el
tangente de cierre y el tramo de cierre desaparecen.  La spline sigue
seleccionada.

Una spline cerrada necesita al menos 3 nodos — pulsar `C` sobre
splines más cortas no hace nada (el editor avisa por el HUD).

---

## 10. Las tres capas de curva

Por cada tramo (la sección de curva entre dos nodos consecutivos) el
editor mantiene tres representaciones independientes:

| Capa | Color | Cuándo se actualiza | Propósito |
|---|---|---|---|
| **Azul** | `#a0a0b8` | Interactiva (cada frame de un arrastre) | Vista previa ágil en tiempo real.  Bézier geodésica híbrida: polígono de control + tramos geodésicos. |
| **Naranja** | `#ff8800` | Workers en segundo plano, ~4–7 s por tramo | Curva "final".  de Casteljau totalmente geodésica, con densificación cascade (fase 2) y *chord-bridging* (fase 3) para que la polilínea pegue de verdad a la superficie incluso en mallas gruesas. |
| **Negra (interp)** | `#000000` | Inmediata, con debounce tras edición | B-spline de scipy a través de los orígenes de los nodos, proyectada a la superficie.  Independiente de los tiradores de tangente — útil cuando quieres una curva que **pase por** los nodos sin más. |

Cada capa se alterna de forma independiente:

- **`b`** — azul on/off
- **`o`** — naranja on/off
- **`k`** — interp (negra) on/off

Los mismos tres estados están reflejados en checkboxes abajo a la
izquierda.

**`r`** — **reconstruye todas las curvas naranja** de cada spline.
Útil tras alternar capas, tras cargar una sesión con muchos tramos, o
tras un crash de algún worker.  El HUD reporta `ORANGE REBUILT` al
terminar.

Mientras la capa naranja se sigue calculando, sus tramos se dibujan
en **naranja más tenue con patrón punteado**.  Cuando un tramo
termina, pasa a naranja sólido brillante.  El HUD muestra progreso
(`COMPUTING ORANGE 12/40`) y `ORANGE DONE` al completarse.

Si el solver de geodésica de un tramo tuvo que recurrir a una recta
(defectos extremos de malla, segmentos entre componentes
desconectadas), ese tramo se repinta en **rojo** y el HUD avisa con
`GEODESIC FALLBACK on span <sid>:<i>` — la curva ahí ya no es
geodésica y deberías o re-rutearla o reparar la malla.

---

## 11. Ayudas visuales

El editor proporciona varios **overlays transitorios** para facilitar
el trabajo de precisión.  Ninguno se guarda en el JSON de sesión;
existen solo durante la edición en vivo.

### Marcador de hover (cursor sobre una curva)

Cuando tu cursor está cerca de cualquier curva visible, aparece un
**visor telescópico** en el punto más cercano de la curva: una
circunferencia fina con una cruz horizontal + vertical, en el color
de la capa de la curva.  La cruz siempre se alinea con los ejes
horizontal / vertical de la pantalla (sensación de mira óptica real
— independiente de la dirección de la curva en 3-D).  Doble clic
izquierdo mientras el marcador está visible inserta un nuevo nodo
exactamente en ese punto.

El marcador se dibuja por encima de la malla (sin z-fighting) pero
solo aparece cuando el punto elegido de la curva es genuinamente
visible desde la cámara — los puntos ocultos detrás de la malla no
reciben marcador.

### Stitch preview (línea gris)

Una fina línea gris conecta constantemente el **último nodo de la
spline activa** con tu **posición del cursor sobre la superficie**.
Es lo que el próximo doble clic adjuntaría.  Cuando el cursor pausa
~150 ms, se autorrefina de una previsualización vertex-snapped
rápida a la geodésica exacta topológicamente insertada — sin
interacción extra.

El stitch desaparece cuando:

- Haces hover sobre un nodo / tirador (una acción distinta ocurriría
  al hacer clic).
- Haces hover sobre una curva (el visor telescópico es ahora el
  punto de inserción relevante).
- La spline activa está cerrada (no tiene sentido "siguiente
  inserción").

### Indicador de snap (dorado / cian)

La esfera dorada (Shift) y la esfera cian (Ctrl) descritas en §8.

### Etiquetas de índice de nodo (mantén `N`)

Mantén la tecla **`N`** — el editor muestra etiquetas numéricas
1-based sobre cada nodo visible.  Una sola spline: solo el índice
del nodo (`3`).  Multi-spline: con el índice 1-based de la spline
como prefijo (`s1:3`).  Las etiquetas:

- Aparecen instantáneamente al pulsar; desaparecen instantáneamente
  al soltar.  Es un atajo **mantén-para-mostrar**, no un toggle.
- Se dibujan en una capa de overlay que ignora la profundidad, así
  que no pueden quedar medio recortadas por la malla.
- Siguen filtradas por **oclusión**: un nodo que esté genuinamente
  al otro lado de la malla no recibe etiqueta, así que solo ves
  números de los nodos que la cámara realmente ve.
- Siguen a los nodos que estés arrastrando.
- Actualizan visibilidad al orbitar la cámara.

### Andamio didáctico (`d`)

Pulsa **`d`** para alternar un andamio de cuatro líneas que
visualiza la cascada de de Casteljau en un parámetro `t` elegido a
lo largo del tramo más reciente de la spline activa.  Aparece un
slider para barrer `t` de 0 a 1 y ver cómo las líneas verdes de
construcción colapsan al punto final sobre la curva.  Útil para
enseñanza, para diagnosticar formas raras y para entender dónde sobre
el tramo cae un `t` dado.  Vuelve a pulsar **`d`** para ocultar el
andamio y el slider.

### Cursor de superficie

Siempre visible.  Un pequeño círculo sobre la malla bajo el ratón,
alineado con el marco local de la superficie.  Te dice dónde caería
un doble clic.

---

## 12. Curvas de guía — referencias auxiliares

A veces la spline que estás dibujando necesita **alinearse** con
curvas que has calculado en otra parte: referencias anatómicas
escaneadas aparte, isofotas de un análisis CAD, anotaciones de
plano, etc.  Impórtalas como **polilíneas de guía**.

### Cargar guías

Pulsa **`Ctrl+X`**.  Se abre un diálogo de selección múltiple
(aceptado: `.vtk` / `.vtp` / `.ply` / `.stl` / `.obj`).  Elige uno o
varios ficheros.  Cada uno se convierte en una polilínea verde
superpuesta sobre la malla.

- El loader extrae solo **celdas de línea**: triángulos u otros
  polígonos en el mismo fichero se descartan silenciosamente, así
  que puedes apuntar a un fichero de malla que contenga también
  líneas de anotación.
- Se aceptan contenedores `vtkPolyData` y `vtkUnstructuredGrid`
  (muchas herramientas escriben datos 1-D como el segundo).  Los
  ficheros `MultiBlock` se desempaquetan a su primer bloque
  relevante.
- El diálogo **reemplaza** las guías cargadas previamente.  Para
  intercambiar un set de guías, pulsa `Ctrl+X` de nuevo y elige una
  selección distinta — no hace falta un "clear" aparte.

### Mantén para previsualizar, suelta para alternar

Pulsa **`x`** (sin Ctrl) para mostrar **temporalmente** todas las
guías en opacidad total mientras la tecla esté pulsada — útil cuando
quieres comprobar la alineación contra una curva sin perder el estilo
fantasma de reposo.  Al **soltar**:

- si las guías estaban *visibles* antes del press → se **ocultan**;
- si las guías estaban *ocultas* antes del press → **quedan visibles**
  y se desvanecen suavemente (~500 ms) desde opacidad total hasta la
  opacidad de reposo `GUIDE_OPACITY`.

Así `x` funciona como toggle *y* como gesto de "echar un vistazo" en
la misma pulsación — un tap se comporta como el toggle clásico, un
hold te da la previsualización, y el release decide el estado final
en cualquier caso.

Si no has cargado ninguna guía, el HUD te lo recuerda con `NO GUIDES
LOADED — use Ctrl+X to import`.

> Cargar un nuevo set de guías (`Ctrl+X`) las arranca siempre
> visibles a la opacidad de reposo, aunque las hubieras ocultado
> antes de importar — se acabó el "las cargué pero la pantalla está
> vacía".

### Estilo

Las guías se renderizan en **verde** con opacidad de reposo
`GUIDE_OPACITY` (por defecto `0.1`, apariencia fantasma para que la
malla subyacente siga siendo legible) y **grosor de línea 3**.  La
duración del fade tras soltar `x` es `GUIDE_FADE_DURATION_SEC = 0.5`.
Las cuatro constantes viven en `SplineConfig` (`GUIDE_COLOR_HEX`,
`GUIDE_LINE_WIDTH`, `GUIDE_OPACITY`, `GUIDE_FADE_DURATION_SEC`) por
si quieres ajustarlas.

### Persistencia

Las guías **no se guardan** en el JSON de sesión — son una
herramienta "mira esto mientras trabajo", no parte de la geometría
de las splines.  Re-impórtalas tras cada carga de sesión.

---

## 13. Opciones de visualización

### Ciclo de opacidad del gizmo (`t`)

Los tiradores (P / A / B) y las líneas de tangente se renderizan
normalmente al **20 % de opacidad** para que no oculten las curvas.
Pulsa **`t`** para ciclar entre `0.2 → 0.4 → 0.7 → 1.0 → 0.2`.  El
HUD reporta el nuevo porcentaje.

Hacer hover sobre cualquier tirador de un nodo bumpea
**temporalmente todo el nodo** (las dos flechas + la línea de
tangente) a opacidad total *y* lo eleva en el z-buffer para que se
dibuje por encima de cualquier curva naranja / azul / negra que
pudiera solaparlo — útil cuando varias splines se cruzan cerca de un
nodo y los tiradores que quieres coger se esconden detrás de una
curva.  Al sacar el cursor el estilo bumpeado persiste durante un
período de gracia de 300 ms antes de revertir (así un giro fugaz del
cursor sobre / fuera del tirador no parpadea), y luego vuelve a la
opacidad del ciclo y al z-depth normal.

### Wireframe (`w`)

Dibuja las aristas de los triángulos de la malla sobre la superficie
sombreada.  Te ayuda a ver dónde aterrizará el snap a vértice /
arista, y a diagnosticar la densidad de la malla.  Pulsa `w` de
nuevo para quitarlo.

### Opacidad de la superficie (`a`)

Cicla la opacidad de la propia malla a través de varios presets
(totalmente opaca / translúcida / muy translúcida).  Útil cuando
necesitas ver splines que pasan por concavidades o detrás de
pliegues.

---

## 14. Guardar, cargar y exportar

### Guardar sesión (`s`)

Pulsa **`s`** para escribir un **JSON** en el directorio actual.  El
nombre es una marca de tiempo (`20260513_184231.json`) en el primer
guardado de la sesión; los guardados siguientes preservan el mismo
nombre base con un sufijo numérico (`..._01.json`) para que el
original nunca se sobreescriba en silencio.

El guardado es **atómico**: se escribe a un hermano `.tmp`, se
hace `fsync`, y se hace `os.replace` sobre el target — nunca ves un
fichero a medio escribir.

### Qué contiene el JSON

- `version` (`2`): la versión del schema.
- `mesh_file`: la ruta o etiqueta de la malla contra la que se
  editó la sesión.  La usan `l` y la CLI para encontrar la
  superficie correcta.
- `splines`: una lista de objetos, cada uno con `closed` (bool) y
  `nodes` (lista).  Cada nodo guarda su `origin` 3-D, los dos
  extremos de tangente `p_a` / `p_b` (o `null` para placeholders), y
  un `id` opcional (1-based, coincide con las etiquetas bajo `N`)
  que el loader ignora — puedes editarlo a mano o borrarlo sin
  consecuencias.

### Cargar sesión (`l`)

Pulsa **`l`** para abrir un diálogo de fichero (por defecto al
`.json` más reciente del directorio actual).  Elige un JSON; el
editor valida su schema, reconstruye cada nodo + tangente + camino,
y lanza los workers de la capa naranja.

Si la validación falla (JSON corrupto, coordenadas NaN, forma
malformada), el editor rechaza la carga y el HUD muestra un error
preciso con línea / columna.  Tu estado actual **no** se muta — las
cargas son todo-o-nada.

### Exportar la curva naranja a VTK (`v`)

Pulsa **`v`** para escribir la curva naranja (totalmente geodésica)
a un fichero `.vtk` binario en el directorio actual.  El número de
muestras lo fija `SplineConfig.EXPORT_VTK_SAMPLES` (por defecto 20
por tramo).  Splines de un solo nodo se escriben como landmarks
`VTK_VERTEX`.  Si la cache naranja en vivo ya tiene el número
correcto de muestras, la exportación la reutiliza (sin recálculo);
si no, recalcula de forma síncrona.

### Exportar caminos geodésicos a TXT (`e`)

Pulsa **`e`** para escribir un fichero de texto estilo CSV con los
caminos geodésicos de cada spline.  Una fila por punto de la
polilínea, con los índices de spline / nodo al lado.

### Exportar por línea de comandos (`spline_export.py`)

Fuera de la GUI, `spline_export.py` consume un JSON de sesión y
imprime / escribe la capa de curva pedida.

```bash
# Por defecto: curva naranja a stdout como CSV.
python spline_export.py sesion.json

# Elegir otra capa (b = azul, o = naranja, k = interp).
python spline_export.py sesion.json b

# Más muestras por tramo (por defecto 60).
python spline_export.py sesion.json o --samples 120

# Escribir a fichero .vtk (mismo nombre base que el JSON).
python spline_export.py sesion.json --vtk

# Escribir a fichero .obj.
python spline_export.py sesion.json --obj

# Usar una malla distinta a la referenciada por el JSON.
python spline_export.py sesion.json hi_res.vtk

# Combinar: malla hi-res + capa interp + muestreo denso.
python spline_export.py sesion.json hi_res.vtk k --samples 200
```

`--obj` y `--vtk` son mutuamente exclusivos.  Sin ninguno, la salida
es CSV por stdout (redirígelo donde quieras).

---

## 15. Resolución de problemas

### "Se quedó colgado después de Loading mesh: …"

Mallas pesadas (cientos de miles de triángulos, además con defectos
no-manifold) hacen que el saneador de topología tarde unos segundos.
Espera — el HUD aparece cuando el saneador termina.  Mallas escala
RVP cargan en 2–3 s; mallas pequeñas / limpias en bastante menos de
un segundo.  Si esperas más de un minuto y no hay progreso, mata con
Ctrl+C y revisa la malla en MeshLab / Blender.

### "Pulsé Ctrl+C y la terminal no se desbloqueó"

No debería pasar en las versiones actuales — antes el runtime
Fortran de Intel MKL interceptaba la interrupción antes de que
Python pudiera limpiar.  Si lo ves a pesar de todo, define la
variable de entorno `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` antes de
lanzar Python:

```bash
# Windows cmd.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
python geo_splines.py …
```

```bash
# Linux / macOS / Git-Bash
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
python geo_splines.py …
```

### "Un tramo naranja está rojo en vez de naranja"

El solver de geodésica tuvo que recurrir a una recta en ese tramo.
Las causas suelen ser un defecto de malla bajo el camino (triángulo
*sliver*, lomo no-manifold, componentes desconectadas).  Inspecciona
la malla en esa región; considera remallarla o mover el nodo
problemático.

### "Las curvas naranja no terminan nunca"

Puede que un worker haya muerto (raro; caso límite de
multiprocessing cross-platform).  Pulsa **`r`** para reconstruir
todas las curvas naranja — eso re-spawnea el pool de workers.

### "Las etiquetas de nodos no aparecen"

La tecla `N` es **mantén-para-mostrar**, no un toggle.  Mantén
pulsado; las etiquetas aparecen al pulsar y desaparecen al soltar.

### "Importé wires.vtk y no se mostró nada"

Si ves `GUIDES LOADED (1 file(s), N segments)` en verde en el HUD
pero no hay líneas verdes en pantalla, las guías pueden estar lejos
del frustum de la cámara.  Pulsa el "reset" de la cámara con clic
medio o orbita afuera — los wires podrían vivir en un espacio de
coordenadas distinto al de la malla cargada.

### "Quiero un marcador de hover más pequeño / grande"

`SplineConfig.HOVER_MARKER_SCREEN_SCALE` (por defecto `0.006`)
escala el radio del visor; los grosores de línea son
`HOVER_MARKER_CIRCLE_LINE_WIDTH` (por defecto 2) y
`HOVER_MARKER_CROSS_LINE_WIDTH` (por defecto 1).  Todas están al
inicio de `geo_splines.py` por si quieres ajustarlas.

---

## 16. Referencia rápida de teclado

### Edición

| Tecla | Acción |
|---|---|
| **Doble clic izq.** sobre la superficie | Añadir nodo a la spline activa |
| **Doble clic izq.** sobre marcador de hover | Insertar nodo en la curva en el punto del hover |
| **Doble clic izq.** sobre nodo de otra spline | Cambiar spline activa |
| **Arrastre P** (esfera roja) | Trasladar nodo |
| **Arrastre A / B** (tiradores) | Rotar tangente (el otro se mantiene simétrico) |
| **Shift + arrastre P** | Snap del nodo al vértice más cercano (indicador dorado) |
| **Ctrl + arrastre P** | Snap del nodo a la arista más cercana (indicador cian) |
| **Shift + arrastre A / B** | Solo magnitud: preserva dirección, cambia longitud |
| **Doble clic der.** sobre superficie | Nueva spline (vacía) / break |
| **Doble clic der.** sobre P | Abre diálogo de edición de coordenadas |
| **Retroceso** | Quitar último nodo o deshacer último "break" |
| **Ctrl + Z** | Deshacer |
| **Ctrl + Y** | Rehacer |
| **C** | Cerrar / reabrir spline activa (≥ 3 nodos para cerrar) |

### Capas de curva + display

| Tecla | Acción |
|---|---|
| **b** | Alternar visibilidad de la capa azul (interactiva) |
| **o** | Alternar visibilidad de la capa naranja (geodésica) |
| **k** | Alternar visibilidad de la capa interp (B-spline) |
| **r** | Reconstruir todas las curvas naranja |
| **t** | Ciclar opacidad del gizmo (20 % → 40 % → 70 % → 100 % → 20 %) |
| **w** | Alternar wireframe de la malla |
| **a** | Ciclar opacidad de la superficie |

### Ayudas visuales

| Tecla | Acción |
|---|---|
| **n** (mantener) | Mostrar etiquetas de índice de nodo mientras se pulsa |
| **d** | Alternar andamio didáctico de de Casteljau |
| **Ctrl + X** | Importar polilíneas de guía (diálogo de fichero).  Se cargan siempre visibles a opacidad de reposo. |
| **x** (mantener) | Mientras está pulsada: guías a opacidad total.  Al soltar: alterna entre oculto y visible (con fade de 500 ms hasta la opacidad de reposo cuando pasa a visible). |

### Sesión + exportación

| Tecla | Acción |
|---|---|
| **s** | Guardar sesión a JSON con marca de tiempo |
| **l** | Cargar sesión desde JSON (diálogo de fichero) |
| **v** | Exportar curva naranja a VTK binario con marca de tiempo |
| **e** | Exportar caminos geodésicos a TXT |

### Cámara

| Acción | Resultado |
|---|---|
| Arrastre izq. (fondo) | Rotar alrededor del punto focal |
| Arrastre medio o Shift+Izq. | Paneo |
| Rueda del ratón | Zoom |
| Arrastre der. | Zoom (o roll, según la build de VTK) |

---

## 17. A dónde ir después

- **Desarrolladores** que quieran extender el editor o entender los
  algoritmos: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — cubre
  kernels JIT, el debounce de master-clock, el pipeline de workers,
  etc.  El schema del JSON está documentado en el
  [`README.md`](README.md).
- **Exportación batch desde CLI**: ver §14 más arriba y
  `spline_export.py --help`.
- **Reportes de bugs / peticiones de feature**: abre un issue con un
  JSON de sesión mínimo + la malla, y el mensaje del HUD en el
  momento en que algo falló.

¡Feliz splining!
