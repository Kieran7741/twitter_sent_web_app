# Twitter Sentiment Flask app

## Getting Started

### Prerequisites

* This application should be run using docker.
* A mongodb image is used to host a local mongodb instance. You should mount a volume to persist data
* Start mongodb instance: docker run --rm -d -p 27017-27019:27017-27019 -v /home/user/data:/data/db --name mongodb mongo
* Log into mongodb container: docker exec -it mongodb bash

## API Documentation

* Found on localhost:5000/api/doc/ : [OpenApi Spec UI](https://localhost:5000/api/doc/)


## Authors

* **Kieran Lyons** - *Initial work* - [Kieran7741](https://github.com/kieran7741)
