# Data-Mining

## Analyse des données

Le CSV contient les données d'identification de la photo, de position, des infos sur la photo (tags, titre), la date de la photo (prise et upload), et trois colonnes nulles.

## Nettoyage des données

On n'a pas besoin de toutes les données du CSV. Par exemple, la seule date qui nous intéresse est la date de prise de vue et pas la date d'upload. Les trois colonnes vides doivent partir.
De plus, certaines données sont dupliquées.
Voici le traitement implémenté : 
- On supprime les lignes où il y a des données dans les colonnes unnamed à la fin
- On enlève les colonnes inutiles
- On enlève les colonnes correspondant à la date d'upload
- On supprime les doublons
- On remplace les champs null dans tags et title par des chaines de caractères vides
- On remplace les 5 champs de dates par une colonne de type Date