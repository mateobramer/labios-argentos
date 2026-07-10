# Correccion asistida clip por clip

Fuente: CHARLA SOBRE EL AMOR Y EL DESAMOR
Split de referencia: vsr_models/splits/val.csv
Cantidad de clips: 233
URL de referencia: https://www.youtube.com/watch?v=_GTu8K8a_Jw

Contexto de la fuente:
- Es una charla larga de Coscu sobre amor/desamor, vida personal, energia, streaming y habitos.
- No confundir con el recorte corto "El mensaje de Coscu sobre EL AMOR" (https://www.youtube.com/watch?v=z-jbSKM49fo), que dura 53 segundos y corresponde a otra fuente del repo.
- Las transcripciones estan normalizadas para dataset: minusculas, sin puntuacion, sin tildes. La salida debe conservar ese formato.

## Tarea

Necesito corregir transcripciones de clips de un dataset de lectura de labios.

Tenes una fuente completa dividida en clips consecutivos. Usa el contexto global para entender palabras raras, pero devolve una correccion clip por clip.

## Reglas estrictas

- No agregues informacion nueva.
- No reescribas estilisticamente.
- No mejores la redaccion si el texto ya se entiende.
- Corregi solo errores evidentes de transcripcion.
- Conserva muletillas, voseo, informalidad, insultos y repeticiones reales.
- Si no estas seguro, no corrijas: marca action="uncertain".
- Si sospechas que podrias estar inventando, marca risk_flags.
- No unas clips.
- No cambies clip_id.
- Devolve solo JSON valido, sin Markdown alrededor.

## Formato de salida obligatorio

{
  "source_id": "CHARLA SOBRE EL AMOR Y EL DESAMOR",
  "corrections": [
    {
      "clip_id": "clip_0000",
      "raw_text": "texto original exacto",
      "suggested_text": "texto sugerido",
      "action": "keep | corrected | uncertain",
      "confidence": 0.0,
      "risk_flags": [],
      "notes": ""
    }
  ]
}

## Reglas de consistencia

- Si action="keep", suggested_text debe ser igual a raw_text.
- Si action="uncertain", suggested_text debe ser igual a raw_text salvo que propongas una alternativa muy clara en notes.
- Si action="corrected", el cambio debe ser minimo y justificable por contexto.
- confidence debe estar entre 0 y 1.
- Tiene que haber exactamente una entrada por clip listado.

## risk_flags posibles

- "nombre_propio"
- "numero"
- "jerga"
- "insulto"
- "muletilla"
- "repeticion"
- "cambio_grande"
- "correccion_inferida"
- "posible_alucinacion"
- "requiere_video"

## Clips ordenados

- clip_0000: voy a antereccionar maria del cerro y ahora que hay un monton de gente en el chat les quiero decir algo
- clip_0001: la vida aca voy a aparecer un hippie o esas finas que leen el horoscopo
- clip_0002: pero la vida posta que se trata de energias
- clip_0003: entienden que estamos haciendo un stream a las 4 de la mañana en españa y tengo 24k de viewers hablando de futbol
- clip_0004: algo totalmente impensado estan saliendo todos los dias los rankings de twitch y a mi
- clip_0005: yo ya tengo 10 11 años de streamer no me interesa ranquear pero te la blanqueo
- clip_0008: ah buena poronga 23k ojo me la sube sabe por que me la sube
- clip_0009: porque eso es mi historia por eso estoy donde estoy por eso soy main event
- clip_0013: ojo yo veo el numero y no les voy a negarme me motivo me motivo
- clip_0014: las cosas estan saliendo bien y se los quiero agradecer si yo tuviera 5k de viewers igual estaria motivado
- clip_0015: igual diria gracias por mirarme
- clip_0016: porque yo agradecido por mas que sea 2 3 4 20 50 500 22000
- clip_0017: agraecido siempre pero vale destacar que en un momento critico porque es un momento critico para mi
- clip_0018: las cosas estan empezando a salir de nuevo despues de que llueve sale el sol
- clip_0019: ojo despues de que llueve sale el sol
- clip_0020: ahora solo me falta terminar el año ganando la velada enamorandome
- clip_0021: y eventualmente el año que viene planeando tener un hijo
- clip_0022: ya esta vida cumplida
- clip_0023: me doy vuelta el juego de la vida no quiero mas nada
- clip_0024: tres hijos tres o dos un chico y una chica un hijo y una hija minimo
- clip_0025: una novia que me ame que me valore que me quiera que me acompañe
- clip_0026: poder acompañarla yo que haga deporte pretensiones bajas no me importa lo demas
- clip_0027: que si el culo que es esto no me importa no me importa
- clip_0028: se tiene un culito lindo mejor mejor
- clip_0029: pero yo prefiero que haga deporte que tenga culo
- clip_0030: el culo un año y te cansas te da igual
- clip_0031: pero que haga deporte ir a verla imaginante que juega el futbol
- clip_0032: que juega el hockey la vas a ver gritar los goles la bancas
- clip_0033: si estas triste estas triste si estas feliz estas feliz con ella
- clip_0034: todos necesitamos amor
- clip_0035: que el amor es lo mas lindo que hay lo mas lindo que hay
- clip_0036: no le tengan miedo al amor no hablan como hice este boludo
- clip_0037: no hablan como hice este boludo yo tenia desde los 16 a los 23 y medio que conocia a mi ex
- clip_0038: desde los 16 a los 23 y medio yo pensaba que la vida era agarchar
- clip_0039: no piensen esa boludez algunos estamos destinados a amar y lo que nos gusta es amar
- clip_0041: y otros estan destinados a ir de flor en flor
- clip_0042: cada uno ve pero no se cierren
- clip_0043: hay que cogerse a todas hay que sacarle punta
- clip_0044: son mentiras hay que experimentar todo hay que probar
- clip_0045: hay que ver donde te sentis mejor
- clip_0046: pero no tengan miedo de enamorarse boludo no tengan miedo de enamorar
- clip_0047: el amor es lo mas lindo que hay asi como es lo mas lindo de lo que mas duele
- clip_0048: la van a pasar mal tambien la van a pasar como el or
- clip_0049: van a llorar van a sufrir van a extrañar por que
- clip_0050: porque lo bueno duele el futbol es hermoso por ejemplo
- clip_0051: una carrera de jugador de futbol lo que tiene de bueno y que todos queremos ser jugadores de futbol
- clip_0052: lo tiene de malo entonces para cada logro importante para cada anhelo importante
- clip_0053: lo importante como es amarte en una pareja en una relacion estable
- clip_0054: va a haber una contracara que es todo lo que tienes que luchar
- clip_0055: todo lo que tienes que sacrificar para poder tenerlo
- clip_0056: lo bueno es dur el objetivo ideal te va a costar
- clip_0057: todo eso que anhelas y soñas si no te cuesta no vale nada
- clip_0058: y si cuesta mucho vale el triple por eso si se enamora metanle
- clip_0059: no importa la edad yo antes pensaba no no tenes 18 no te enamores
- clip_0060: enamorarte lo 18 te quieres enamorar lo 18 no no no
- clip_0061: te enamoras no es que te queres porque uno no elige te enamoras lo 18
- clip_0062: mandale mandale cumbia
- clip_0063: que te vas a mentir a vos mismo de que no estas enamorado
- clip_0064: de que no queres amar estas loco
- clip_0065: te puedes enamorar de cualquier edad yo pensaba que no yo pensaba que no
- clip_0066: yo pensaba no no no veo esta piba y listo
- clip_0067: no la veo mas mira si me engancho y no puedo estar con tal otra
- clip_0068: que haces que haces para que te sirve estar con este y con esta
- clip_0069: y con este y con esta esa es la que nos cuentan que te cuentan para ser un macho alfa
- clip_0070: tiene que estar con 5 6 7 8 es mentira
- clip_0071: y a mi me encanta garchar como cualquier persona pero amigo estar de novio es otro level
- clip_0072: estar de novio es otro level despertarse todas las mañanas
- clip_0074: eso es otro punto level amigo tu casilla llena de mensajes
- clip_0075: ojala no porque esas cosas no pasan pero tu casilla tiene 7 mensajes
- clip_0076: vos queres leer uno solo a vos te importa uno solo
- clip_0077: cuando empezas la relacion a mi es la etapa de oro
- clip_0078: el primer año de relacion es la etapa de oro
- clip_0079: solo queres hablar con ella no la ves un dia la extrañas
- clip_0080: se llega a ir de viaje y vas a estar todos los dias todos los dias
- clip_0081: comiendote las uñas intentando no ser pesado pero estar para ella
- clip_0082: llamandola pero sin romperle las pelotas y sin cagarle el viaje
- clip_0083: te vuelves loco para bien loco te enamoras
- clip_0085: despues yo tengo una teoria de que las relaciones en general
- clip_0086: se desgastan y que uno no puede hacer nada mas que ir mutando la relacion
- clip_0087: y convirtiendola en lo que tenga que ser para que sobreviva
- clip_0089: lo que se disfruta mucho humo dice uno
- clip_0090: mucho humo no mucho humo no
- clip_0091: te lo digo de corazon cierro la charla del amor para eso el amor es lo mas lindo que hay
- clip_0092: mi orden de cosas es dormir
- clip_0093: hacer el amor comer cagar y coger
- clip_0094: porque hacer el amor y coger no es lo mismo
- clip_0096: si vos no amas a la otra persona no vas a poder hacer el del amor se lo juro
- clip_0097: bueno segun mi teoria eh segun mi teoria hay gente que dice
- clip_0098: yo puedo hacer el amor aunque aunque yo puedo hacer el amor con alguien
- clip_0099: aunque sea la primera vez que lo veo mentira
- clip_0100: para mi mentira para hacer el amor tenes que sentir amor
- clip_0101: imposible si no por eso en mi ranking lo vuelvo a decir
- clip_0102: cagar vale mas que coger cagar vale mas que coger
- clip_0103: coger esta super sobrevalorado hacer el amor es otro level
- clip_0104: eso si que vale eso si que vale y la paja uff es que la paja
- clip_0105: la paja top 3 si o si la paja top 3 si o si
- clip_0106: y lo ultimo que voy a decir porque hay muchos que estan poniendo que despues te rompen el corazon
- clip_0107: y que el amor trae una catastrofe
- clip_0108: sabes que pasa otro de mis mejores momentos fue cuando me rompieron el corazon
- clip_0109: a mi me rompieron el corazon y yo tuve otra de las mejores epocas de mi vida
- clip_0110: que saben cual fue la de recuperacion la de superacion
- clip_0111: la de amarse a uno mismo
- clip_0112: os amas amas amas amas amas amas de golpe esa persona
- clip_0113: se va se va
- clip_0114: y todo ese amor que vos estabamos dando
- clip_0115: hace asi no sabe a donde ir no sabe a donde ir
- clip_0116: porque vos lo queres dar no te lo queres dar vos mismo vos te odias vos te odias
- clip_0118: y empezas por que se fue
- clip_0119: como no valora como no
- clip_0120: como no me quiere como no se quiere quedar conmigo
- clip_0121: como no como no esta aca
- clip_0122: entonces todo ese amor que vos dabas y dabas y dabas todo ese amor
- clip_0123: la interferencia todo ese amor que vos dabas y dabas y dabas no va a ningun lado
- clip_0124: porque ya no hay nadie aqui a andar de amor pero que pasa
- clip_0125: depresion estas mal no sali de tu casa
- clip_0126: tenes barba te queres en las pelas en la chota
- clip_0127: normal te sentis triste te sentis mal
- clip_0128: hasta aqui un dia haces el clic
- clip_0129: y eso es lo importante de cuando termine una relacion mucho me viene a preguntar aca
- clip_0130: corte con mi novia estoy remal se que es una catastrofe pero despues de la catastrofe
- clip_0131: viene una etapa increible para nosotros
- clip_0132: saben cual es cuando hacemos el duelo
- clip_0133: cuando nos tomamos nuestro tiempo cuando lloramos lloramos lloramos
- clip_0134: y de golpe nos damos cuenta que todo ese amor
- clip_0135: que estabamos dando sigue ahi lo agarras lo agarras lo agarras
- clip_0136: lo agarras lo agarras lo agarras y te lo metes adentro
- clip_0137: y sabes que haces te juntas con los pibias te pones el bol corre corre corre
- clip_0138: vas al gimnasio te pones mejor te cuidas con las comidas
- clip_0139: empiezas a tomar un poco menos alcohol empiezas a dejar de voludiar saliendo todos los dias
- clip_0140: salir es una paron esta sobrevaloradisimo salir te empiezas a juntar con amigos
- clip_0141: empiezas a repartir ese amor en toda la gente que sigue ahi
- clip_0142: porque vos tenes una novia y yo te banco
- clip_0143: la novia te quita sin querer tiempo para tus amigos
- clip_0144: la novia te quita sin querer tiempo para tu vida
- clip_0145: para el gimnasio para no se jugar al futbol para hacer tu hobby lo que te gusta te lo quita
- clip_0146: por que porque vos queres pasar tiempo con ella esta buenisimo
- clip_0147: pero cuando esa persona se va se reparte la baraja de nuevo
- clip_0148: con una persona menos se fue tu amor hay una carta mas
- clip_0150: en todos los pibes que siguen ahi en todos tus amigos que te siguen mancando
- clip_0151: y en vez de decirle no no puedo ir a comer no no puedo ir a comer jorgeito
- clip_0152: porque tengo que quedarme viendo una peli con mi novia
- clip_0153: vamos a comer de nuevo vamos a comer
- clip_0154: vamos a comer dale vamos a comer amigo posta una de las mejores etapas
- clip_0155: que tuve yo fue cuando termine mi relacion y tenia tiempo
- clip_0156: para aprender todos los dias obviamente yo estuve muy mal cuando termine mi relacion
- clip_0157: mi somierda mi somierda me desoriento yo proyectaba
- clip_0158: todo ahi porque bueno viste pero una persona como yo que nunca tuvo novia
- clip_0159: de igual que siente que hay una sola persona en el mundo que vas a amar ojala no sea asi
- clip_0160: pues yo eso pensaba yo entonces yo literalmente
- clip_0161: dije se acabo el mundo para mi literal se acabo el mundo para mi
- clip_0162: no va a haber otra igual no va a haber nunca otra persona igual no voy a conseguir
- clip_0163: nunca una persona asi nunca y pensaba eso y pensaba eso
- clip_0164: y pensaba eso y lloraba y hablaba con mis amigos que le mando un saludo a los chicos de quino de combo
- clip_0165: que me habran fumado de no ya soy viejo no voy a conseguir
- clip_0166: otra persona etcetera etcetera etcetera amigo
- clip_0167: se termina y te puedo asegurar
- clip_0168: que sufris sufris sufris un dia basta un dia te cansas
- clip_0169: y ahi te convertis en supras alla
- clip_0170: los hombres del chat me van a entender
- clip_0171: bien en la fase 2 la fase 2 cual es
- clip_0173: que me gustan si deje la facultad si me falta unas materias la reanudo
- clip_0174: por que porque la vida sigue y sabes que teneis que hacer cuando la vida sigue
- clip_0175: sobrevivir que sobrevivir ponerte las pilas
- clip_0176: en todo y volver a tener
- clip_0177: una vida social y eventualmente volver a conseguir al amor de tu vida
- clip_0178: como se consigue al amor de tu vida estando hecho mierda con barba todo abandonado
- clip_0179: o poniendote las pilas con tu vida y haciendote cargo obviamente
- clip_0180: haciendote cargo entonces cuando te viene ese pensamiento
- clip_0181: de que la vida sigue y de que ese amor se fue todo es nuevo volves a estar alla arriba
- clip_0183: meti al gim como nunca empece una dieta nueva
- clip_0184: no tengas miedo no tengas miedo se termina una relacion
- clip_0185: es una paja duela un monton pero termina
- clip_0187: la vida esta llena de quilombos
- clip_0188: que te generan una fortaleza a la larga
- clip_0189: perdiste una relacion vendra otra perdiste una relacion fortaleciste
- clip_0190: todas las que te quedaron perdiste un amigo porque no es solo perder una novia
- clip_0191: a veces perdes un amigo fortaleces cuatro amigos
- clip_0192: que siguen ahi y que te estan bancando uno se mando una cagada
- clip_0193: nunca mas gil pero hay cuatro que siguen siendo hermanos
- clip_0194: y a esos hermanos todo
- clip_0195: charlasad no no charla motivadora loco bueno gente
- clip_0196: vamos a ver el tema maria de serra queria que sepan
- clip_0197: que me gusta hablar con ustedes que me gusta darles un consejo
- clip_0198: y yo ya se que parezco joven porque me cuido bien
- clip_0199: y porque no tomo alcohol que tambien la recomiendo que no sean pelotudos no tomen alcohol
- clip_0200: el alcohol es una mierda y el alcohol
- clip_0201: es una mierda yo ustedes saben lo que quieran porque es su cuerpo
- clip_0202: su salud pero yo nunca nada
- clip_0203: solo cuando salio campeon a seleccion un ferneo y algunas cositas
- clip_0204: pero nunca nada porro no tampoco
- clip_0205: el porro esta bien dice uno en el chat para mi el porro no esta bien
- clip_0206: perdon disculpeme para mi queres fumar un porro te banco
- clip_0207: te respeto y para mi
- clip_0208: en mi vida el porro no esta bien no me importa si es natural
- clip_0209: yo no quiero porro cerca no me gusta el porro
- clip_0210: no me gusta el porro y lo digo de una
- clip_0211: no tengo que quedar bien con nadie yo les digo lo que yo siento
- clip_0212: si mi mejor amigo fuma porro nunca le voy a decir nada nunca le voy a decir che
- clip_0213: que haces con esa mierda tu cuerpo tu salud te respeto
- clip_0214: esta todo viola yo no voy a fumar en mi vida voy a fumar
- clip_0215: dijo porno no el porno es lo mismo que me gusta el porno lo banco el porro no banco
- clip_0216: no me gusta el porro la marihuana eso no me gusta
- clip_0217: nunca me gusto nunca senti que la necesites
- clip_0218: para mi te lleva a un lugar donde
- clip_0219: no sos vos donde
- clip_0220: no se no sos vos
- clip_0221: pero bueno no quiero entrar en esa discusion porque es personal es personal nunca no vamos a poner de acuerdo
- clip_0222: nunca no vamos a poner de acuerdo de unico que les puedo agregar es
- clip_0223: yo tomo alcohol para desinhibirme amigo y si mejor no aprendes
- clip_0224: a desinhibirte sin la necesidad de alcohol y si mejor no aprendes a soltarte
- clip_0225: sin necesitar tomar nada en mi opinion eh
- clip_0226: arrancas a los djs sos timido te cuesta a los djs
- clip_0227: si no tomo alcohol no me encargo una mina bueno el problema
- clip_0228: esta en tu seguridad
- clip_0229: que tenes que meterte alcohol para encararte una mina yo no te van
- clip_0230: yo sabre que te voy a decir andad a poco intentadlo de a poco date tiempo
- clip_0231: a los 16 años por ahi no estas listo para encararte una persona es dificil encarar
- clip_0232: a todos nos cuesta encarar sobrio
- clip_0233: mas dificil todavia pero yo
- clip_0234: nunca te voy a decir entonces anda ponerte en pedo
- clip_0235: y encarar porque esa no es la solucion en la vida no vas a estar toda la vida en pedo
- clip_0236: para poder solucionar tus problemas porque tal vez estoy en pedo
- clip_0237: no en la vida vas a estar la mayor parte de tu vida sobrio y vas a tener que aprender
- clip_0238: a resolver los problemas sobrio entonces por que mejor no mejoras tu lado sobrio
- clip_0239: antes de trabajar en el lado ebrio si quieren tomar
- clip_0240: tomen con precaucion si quieren fumar fumen con precaucion yo no me voy a meter
- clip_0241: en la vida de nadie simplemente les estoy diciendo que es lo que hago yo
- clip_0242: que consumo yo que no consumo yo yo consumo vegetales ahora bueno por la velada
- clip_0243: carnes pero despues voy a volver a consumir vegetales porque me encanta ser vegetariano
- clip_0244: consumo vegetales consumo agua
- clip_0245: y consumo cero drogas cero alcohol y me encanta porque tengo 31 años
- clip_0246: y parezco de 28
- clip_0247: yo creo sin el video de 28 de 40 nada eso no entro

## Texto continuo de contexto

voy a antereccionar maria del cerro y ahora que hay un monton de gente en el chat les quiero decir algo la vida aca voy a aparecer un hippie o esas finas que leen el horoscopo pero la vida posta que se trata de energias entienden que estamos haciendo un stream a las 4 de la mañana en españa y tengo 24k de viewers hablando de futbol algo totalmente impensado estan saliendo todos los dias los rankings de twitch y a mi yo ya tengo 10 11 años de streamer no me interesa ranquear pero te la blanqueo ah buena poronga 23k ojo me la sube sabe por que me la sube porque eso es mi historia por eso estoy donde estoy por eso soy main event ojo yo veo el numero y no les voy a negarme me motivo me motivo las cosas estan saliendo bien y se los quiero agradecer si yo tuviera 5k de viewers igual estaria motivado igual diria gracias por mirarme porque yo agradecido por mas que sea 2 3 4 20 50 500 22000 agraecido siempre pero vale destacar que en un momento critico porque es un momento critico para mi las cosas estan empezando a salir de nuevo despues de que llueve sale el sol ojo despues de que llueve sale el sol ahora solo me falta terminar el año ganando la velada enamorandome y eventualmente el año que viene planeando tener un hijo ya esta vida cumplida me doy vuelta el juego de la vida no quiero mas nada tres hijos tres o dos un chico y una chica un hijo y una hija minimo una novia que me ame que me valore que me quiera que me acompañe poder acompañarla yo que haga deporte pretensiones bajas no me importa lo demas que si el culo que es esto no me importa no me importa se tiene un culito lindo mejor mejor pero yo prefiero que haga deporte que tenga culo el culo un año y te cansas te da igual pero que haga deporte ir a verla imaginante que juega el futbol que juega el hockey la vas a ver gritar los goles la bancas si estas triste estas triste si estas feliz estas feliz con ella todos necesitamos amor que el amor es lo mas lindo que hay lo mas lindo que hay no le tengan miedo al amor no hablan como hice este boludo no hablan como hice este boludo yo tenia desde los 16 a los 23 y medio que conocia a mi ex desde los 16 a los 23 y medio yo pensaba que la vida era agarchar no piensen esa boludez algunos estamos destinados a amar y lo que nos gusta es amar y otros estan destinados a ir de flor en flor cada uno ve pero no se cierren hay que cogerse a todas hay que sacarle punta son mentiras hay que experimentar todo hay que probar hay que ver donde te sentis mejor pero no tengan miedo de enamorarse boludo no tengan miedo de enamorar el amor es lo mas lindo que hay asi como es lo mas lindo de lo que mas duele la van a pasar mal tambien la van a pasar como el or van a llorar van a sufrir van a extrañar por que porque lo bueno duele el futbol es hermoso por ejemplo una carrera de jugador de futbol lo que tiene de bueno y que todos queremos ser jugadores de futbol lo tiene de malo entonces para cada logro importante para cada anhelo importante lo importante como es amarte en una pareja en una relacion estable va a haber una contracara que es todo lo que tienes que luchar todo lo que tienes que sacrificar para poder tenerlo lo bueno es dur el objetivo ideal te va a costar todo eso que anhelas y soñas si no te cuesta no vale nada y si cuesta mucho vale el triple por eso si se enamora metanle no importa la edad yo antes pensaba no no tenes 18 no te enamores enamorarte lo 18 te quieres enamorar lo 18 no no no te enamoras no es que te queres porque uno no elige te enamoras lo 18 mandale mandale cumbia que te vas a mentir a vos mismo de que no estas enamorado de que no queres amar estas loco te puedes enamorar de cualquier edad yo pensaba que no yo pensaba que no yo pensaba no no no veo esta piba y listo no la veo mas mira si me engancho y no puedo estar con tal otra que haces que haces para que te sirve estar con este y con esta y con este y con esta esa es la que nos cuentan que te cuentan para ser un macho alfa tiene que estar con 5 6 7 8 es mentira y a mi me encanta garchar como cualquier persona pero amigo estar de novio es otro level estar de novio es otro level despertarse todas las mañanas eso es otro punto level amigo tu casilla llena de mensajes ojala no porque esas cosas no pasan pero tu casilla tiene 7 mensajes vos queres leer uno solo a vos te importa uno solo cuando empezas la relacion a mi es la etapa de oro el primer año de relacion es la etapa de oro solo queres hablar con ella no la ves un dia la extrañas se llega a ir de viaje y vas a estar todos los dias todos los dias comiendote las uñas intentando no ser pesado pero estar para ella llamandola pero sin romperle las pelotas y sin cagarle el viaje te vuelves loco para bien loco te enamoras despues yo tengo una teoria de que las relaciones en general se desgastan y que uno no puede hacer nada mas que ir mutando la relacion y convirtiendola en lo que tenga que ser para que sobreviva lo que se disfruta mucho humo dice uno mucho humo no mucho humo no te lo digo de corazon cierro la charla del amor para eso el amor es lo mas lindo que hay mi orden de cosas es dormir hacer el amor comer cagar y coger porque hacer el amor y coger no es lo mismo si vos no amas a la otra persona no vas a poder hacer el del amor se lo juro bueno segun mi teoria eh segun mi teoria hay gente que dice yo puedo hacer el amor aunque aunque yo puedo hacer el amor con alguien aunque sea la primera vez que lo veo mentira para mi mentira para hacer el amor tenes que sentir amor imposible si no por eso en mi ranking lo vuelvo a decir cagar vale mas que coger cagar vale mas que coger coger esta super sobrevalorado hacer el amor es otro level eso si que vale eso si que vale y la paja uff es que la paja la paja top 3 si o si la paja top 3 si o si y lo ultimo que voy a decir porque hay muchos que estan poniendo que despues te rompen el corazon y que el amor trae una catastrofe sabes que pasa otro de mis mejores momentos fue cuando me rompieron el corazon a mi me rompieron el corazon y yo tuve otra de las mejores epocas de mi vida que saben cual fue la de recuperacion la de superacion la de amarse a uno mismo os amas amas amas amas amas amas de golpe esa persona se va se va y todo ese amor que vos estabamos dando hace asi no sabe a donde ir no sabe a donde ir porque vos lo queres dar no te lo queres dar vos mismo vos te odias vos te odias y empezas por que se fue como no valora como no como no me quiere como no se quiere quedar conmigo como no como no esta aca entonces todo ese amor que vos dabas y dabas y dabas todo ese amor la interferencia todo ese amor que vos dabas y dabas y dabas no va a ningun lado porque ya no hay nadie aqui a andar de amor pero que pasa depresion estas mal no sali de tu casa tenes barba te queres en las pelas en la chota normal te sentis triste te sentis mal hasta aqui un dia haces el clic y eso es lo importante de cuando termine una relacion mucho me viene a preguntar aca corte con mi novia estoy remal se que es una catastrofe pero despues de la catastrofe viene una etapa increible para nosotros saben cual es cuando hacemos el duelo cuando nos tomamos nuestro tiempo cuando lloramos lloramos lloramos y de golpe nos damos cuenta que todo ese amor que estabamos dando sigue ahi lo agarras lo agarras lo agarras lo agarras lo agarras lo agarras y te lo metes adentro y sabes que haces te juntas con los pibias te pones el bol corre corre corre vas al gimnasio te pones mejor te cuidas con las comidas empiezas a tomar un poco menos alcohol empiezas a dejar de voludiar saliendo todos los dias salir es una paron esta sobrevaloradisimo salir te empiezas a juntar con amigos empiezas a repartir ese amor en toda la gente que sigue ahi porque vos tenes una novia y yo te banco la novia te quita sin querer tiempo para tus amigos la novia te quita sin querer tiempo para tu vida para el gimnasio para no se jugar al futbol para hacer tu hobby lo que te gusta te lo quita por que porque vos queres pasar tiempo con ella esta buenisimo pero cuando esa persona se va se reparte la baraja de nuevo con una persona menos se fue tu amor hay una carta mas en todos los pibes que siguen ahi en todos tus amigos que te siguen mancando y en vez de decirle no no puedo ir a comer no no puedo ir a comer jorgeito porque tengo que quedarme viendo una peli con mi novia vamos a comer de nuevo vamos a comer vamos a comer dale vamos a comer amigo posta una de las mejores etapas que tuve yo fue cuando termine mi relacion y tenia tiempo para aprender todos los dias obviamente yo estuve muy mal cuando termine mi relacion mi somierda mi somierda me desoriento yo proyectaba todo ahi porque bueno viste pero una persona como yo que nunca tuvo novia de igual que siente que hay una sola persona en el mundo que vas a amar ojala no sea asi pues yo eso pensaba yo entonces yo literalmente dije se acabo el mundo para mi literal se acabo el mundo para mi no va a haber otra igual no va a haber nunca otra persona igual no voy a conseguir nunca una persona asi nunca y pensaba eso y pensaba eso y pensaba eso y lloraba y hablaba con mis amigos que le mando un saludo a los chicos de quino de combo que me habran fumado de no ya soy viejo no voy a conseguir otra persona etcetera etcetera etcetera amigo se termina y te puedo asegurar que sufris sufris sufris un dia basta un dia te cansas y ahi te convertis en supras alla los hombres del chat me van a entender bien en la fase 2 la fase 2 cual es que me gustan si deje la facultad si me falta unas materias la reanudo por que porque la vida sigue y sabes que teneis que hacer cuando la vida sigue sobrevivir que sobrevivir ponerte las pilas en todo y volver a tener una vida social y eventualmente volver a conseguir al amor de tu vida como se consigue al amor de tu vida estando hecho mierda con barba todo abandonado o poniendote las pilas con tu vida y haciendote cargo obviamente haciendote cargo entonces cuando te viene ese pensamiento de que la vida sigue y de que ese amor se fue todo es nuevo volves a estar alla arriba meti al gim como nunca empece una dieta nueva no tengas miedo no tengas miedo se termina una relacion es una paja duela un monton pero termina la vida esta llena de quilombos que te generan una fortaleza a la larga perdiste una relacion vendra otra perdiste una relacion fortaleciste todas las que te quedaron perdiste un amigo porque no es solo perder una novia a veces perdes un amigo fortaleces cuatro amigos que siguen ahi y que te estan bancando uno se mando una cagada nunca mas gil pero hay cuatro que siguen siendo hermanos y a esos hermanos todo charlasad no no charla motivadora loco bueno gente vamos a ver el tema maria de serra queria que sepan que me gusta hablar con ustedes que me gusta darles un consejo y yo ya se que parezco joven porque me cuido bien y porque no tomo alcohol que tambien la recomiendo que no sean pelotudos no tomen alcohol el alcohol es una mierda y el alcohol es una mierda yo ustedes saben lo que quieran porque es su cuerpo su salud pero yo nunca nada solo cuando salio campeon a seleccion un ferneo y algunas cositas pero nunca nada porro no tampoco el porro esta bien dice uno en el chat para mi el porro no esta bien perdon disculpeme para mi queres fumar un porro te banco te respeto y para mi en mi vida el porro no esta bien no me importa si es natural yo no quiero porro cerca no me gusta el porro no me gusta el porro y lo digo de una no tengo que quedar bien con nadie yo les digo lo que yo siento si mi mejor amigo fuma porro nunca le voy a decir nada nunca le voy a decir che que haces con esa mierda tu cuerpo tu salud te respeto esta todo viola yo no voy a fumar en mi vida voy a fumar dijo porno no el porno es lo mismo que me gusta el porno lo banco el porro no banco no me gusta el porro la marihuana eso no me gusta nunca me gusto nunca senti que la necesites para mi te lleva a un lugar donde no sos vos donde no se no sos vos pero bueno no quiero entrar en esa discusion porque es personal es personal nunca no vamos a poner de acuerdo nunca no vamos a poner de acuerdo de unico que les puedo agregar es yo tomo alcohol para desinhibirme amigo y si mejor no aprendes a desinhibirte sin la necesidad de alcohol y si mejor no aprendes a soltarte sin necesitar tomar nada en mi opinion eh arrancas a los djs sos timido te cuesta a los djs si no tomo alcohol no me encargo una mina bueno el problema esta en tu seguridad que tenes que meterte alcohol para encararte una mina yo no te van yo sabre que te voy a decir andad a poco intentadlo de a poco date tiempo a los 16 años por ahi no estas listo para encararte una persona es dificil encarar a todos nos cuesta encarar sobrio mas dificil todavia pero yo nunca te voy a decir entonces anda ponerte en pedo y encarar porque esa no es la solucion en la vida no vas a estar toda la vida en pedo para poder solucionar tus problemas porque tal vez estoy en pedo no en la vida vas a estar la mayor parte de tu vida sobrio y vas a tener que aprender a resolver los problemas sobrio entonces por que mejor no mejoras tu lado sobrio antes de trabajar en el lado ebrio si quieren tomar tomen con precaucion si quieren fumar fumen con precaucion yo no me voy a meter en la vida de nadie simplemente les estoy diciendo que es lo que hago yo que consumo yo que no consumo yo yo consumo vegetales ahora bueno por la velada carnes pero despues voy a volver a consumir vegetales porque me encanta ser vegetariano consumo vegetales consumo agua y consumo cero drogas cero alcohol y me encanta porque tengo 31 años y parezco de 28 yo creo sin el video de 28 de 40 nada eso no entro
