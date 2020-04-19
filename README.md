# Twitter Sentiment Flask app

## Getting Started
For local development

### Prerequisites

* This application should be run using docker.
* A mongodb image is used to host a local mongodb instance. You should mount a volume to persist data
* Start mongodb instance:
 ```commandline
 docker run --rm -d -p 27017-27019:27017-27019 -v <absolute_path_to_local_folder>:/data/db --name mongodb mongo

Example:
docker run --rm -d -p 27017-27019:27017-27019 -v /Users/kieran/Documents/code/twitter_sentiment/data/:/data/db --name mongodb mongo
```
* Log into mongodb container:
```commandline
docker exec -it mongodb bash
``` 
* Stop container:
```commandline
docker stop mongodb
```

## API Documentation

* Found on localhost:5000/api/doc/ : [OpenApi Spec UI](https://localhost:5000/api/doc/)


## Authors

* **Kieran Lyons** - *Initial work* - [Kieran7741](https://github.com/kieran7741)
