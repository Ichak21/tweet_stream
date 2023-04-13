# tweets_stream

## Installer vos images et containers dans Docker

- Placez vous dans le directory où se situe votre fichier docker-compose.yaml
  - soit ../kafka/
  - soit ../cassandra/
- lancer la commande `docker compose up -d` afin de lancer vos containers et images dans Docker.

Cela permettra d'insttaller les images/containers sur Docker en se pbasantr sur la configuration des fichiers docker-compose.yaml

Il se peut que vous ayez à faire des modifications dans le fichier cassandra/docker-compose.yaml :
- La partie volumétrie dans les nœuds Cassandra peut être commentée.
- Sinon remplacer le par le chemin où vous souhaitez stocker vos données de volumétrie Cassandra
  - ![image](https://user-images.githubusercontent.com/75131876/231718502-5073938b-07c9-4420-a5d4-c89ee8e6effb.png)
  - your\path\:/var/lib/cassandra

## Installer le model
- Telecharger le model : 
- Mettre le fichier *model_rfc.joblib* dans /kafka/AI_model
  
## Lancer le Producer et le Consumer

- Ouvrir 2 terminaux (Producer & Consumer)
- Se positionner sur le dir tweet_stream/kafka/ dans les 2 terminaux
  - ![image](https://user-images.githubusercontent.com/75131876/231719432-58d3ef1c-1310-4a47-b76e-ca957eba794f.png)
- lancer les scripts python pour exécuter le Producer & Consumer afin de streamer les tweets et les consommer via le Consumer (Python) pour le preprocessing, mais également l'ajout dans les tables Cassandra.
  - commande shell `python .\kafka_producer.py`
  - commande shell `python .\kafka_consumer.py`




