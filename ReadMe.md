# AI-Playground

Das ist nur ein Playground und es ist nicht zwingen sinnvoll ein NN für die Aufgaben zu verwenden.

### Ansätze:
- Mit Tensorflow versuchen schnell und einfach ein Ergebnis zu erziehlen, siehe [Tensorflow-Ordner](https://github.com/b2aff6009/ai-playground/tree/main/tensorflow)
- Von Grund auf ein NN selbst implementieren, mit allem was dazu gehört. Siehe [self-constructed-Ordner](https://github.com/b2aff6009/ai-playground/tree/main/self-constructed)


## Erstes Projekt (sorted)
Mit beiden Ansätzen wollte ich als erstes ein NN bauen, welches Prüft ob die Eingabe sortiert ist oder nicht. 
- Eingabe: Liste an Zahlen (float)
- Ausgabe: 0 bis 1, wobei nur 1.0 bedeutet, dass die Eingabe sortiert ist.

### Tensorflow 
 - File: [sorted.py](https://github.com/b2aff6009/ai-playground/blob/main/tensorflow/sorted.py)
 - Hier wird in Zeile 44-48 ein Model erzeugt, dass paar Layer hat. Später in Zeile 53 dann mit Daten trainiert wird. Und in 56 wird dann eine Vorhersage getroffen.
 Ausgabe:
``` 
    Value0  Value1  Value2  Value3  Value4  Value5  Value6  Value7  Value8  Value9
13       0       1       2       3       4       5       6       7       8       9
7        8       6       9       4      10      11      12       7      13       5
18       3       4       5       6       7       8       9      10      11      12
8        0       1       2       3       4       5       6       7       8       9
[1, 0, 1, 1]
```
Wobei die letzte Zeile enthält ob die oberen Daten (Zeilenweise) sortiert sind.

### Self-constructed
- File: [sorted.py](https://github.com/b2aff6009/ai-playground/blob/main/self-constructed/sorted.py)

Mit dem folgenden Code wir ein Netz mit 4 Layern erzeugt. Die Anzahl der Eingabe nodes hängt von der größe des Eingegebenen-Arrays ab:
```
	net = nn.Net(inputLen, costFunction)
	net.setInput(inputs[0]["values"])

	net.addLayer(nn.NeightboursLayer(net, inputLen-1, 0.5, nn.relu, net.layerCnt(), 0.5, [-1, 1])) #get diff
	net.addLayer(nn.NeightboursLayer(net, inputLen-1, -0.5, nn.sigmoid, net.layerCnt(), 0.5, [10])) # normalize
	net.addLayer(nn.DenseLayer(net, 1, 0.0, nn.relu, net.layerCnt(), -0.5 + inputLen/2, [1]*(inputLen-1))) # sum up
	net.addLayer(nn.NeightboursLayer(net, 1, 0.0, nn.relu, net.layerCnt(), -0.5 + inputLen/2, [1/(inputLen-1)])) # normalize 
```

#### Manuel konfiguriertes Netz
Hier zwei Beispiele beidenen das Netz nicht trainiert ist, sondern bei dem Ich alle Gewichtungen selbst eingegeben habe. 
- [sortierte Eingabe](https://raw.githubusercontent.com/b2aff6009/ai-playground/main/outputs/manual_sort_net_2.svg)
- [unsortierte Eingabe](https://raw.githubusercontent.com/b2aff6009/ai-playground/main/outputs/manual_sort_net.svg)

Die Gewichtungen, wie stark eine Verbindung zwischen zwei Nodes ist steht im Tooltip wenn man über die Verbindung hooverd.

#### Trainiertes Netz
Hier sitzt ich gerade dran.