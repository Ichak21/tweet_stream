# tweets_stream

## Installer vos images et containers dans Docker

- Placez vous dans le directory où se situe votre fichier docker-compose.yaml
  - soit ../kafka/
  - soit ../cassandra/
- lancer la commande `docker compose up -d` afin de lancer vos containers et images dans Docker.

Cela permettra d'installer les images/containers sur Docker en se basant sur la configuration des fichiers docker-compose.yaml

Il se peut que vous ayez à faire des modifications dans le fichier cassandra/docker-compose.yaml :
- La partie volumétrie dans les nœuds Cassandra peut être commentée.
- Sinon remplacez-le par le chemin où vous souhaitez stocker vos données de volumétrie Cassandra
  - ![image](https://user-images.githubusercontent.com/75131876/231718502-5073938b-07c9-4420-a5d4-c89ee8e6effb.png)
  - your\path\:/var/lib/cassandra

## Installer le model
- Télécharger le model : https://drive.google.com/file/d/1Ee9whVScl8eXnEaTD9SORG0IeLN63WH5/view?usp=sharing
- Mettre le fichier *model_rfc.joblib* dans /kafka/AI_model
- Pour lancer l'entraînement du model il faudra planifier une tâche sur : /kafka/AI_model/model_entrainement.py
Attention : le consumer doit redémarrer pour prendre en compte le nouveau model
  
## Lancer le Producer et le Consumer

- Ouvrir 2 terminaux (Producer & Consumer)
- Se positionner sur le dir tweet_stream/kafka/ dans les 2 terminaux
  - ![image](https://user-images.githubusercontent.com/75131876/231719432-58d3ef1c-1310-4a47-b76e-ca957eba794f.png)
- lancer les scripts python pour exécuter le Producer & Consumer afin de streamer les tweets et les consommer via le Consumer (Python) pour le preprocessing, mais également l'ajout dans les tables Cassandra.
  - commande shell `python .\kafka_producer.py`
  - commande shell `python .\kafka_consumer.py`




