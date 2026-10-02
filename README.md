## MongoDB run storage

The API stores one document in the `runs` collection after every successful
`POST /api/v1/agent/chat` execution. It uses PyMongo's native
`AsyncMongoClient`, which is the current MongoDB async Python driver API.

Configure the connection in `.env`:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=token_compression
MONGODB_RUNS_COLLECTION=runs
```

For a local MongoDB instance, run:

```bash
docker run --name token-compression-mongo --publish 27017:27017 --detach mongo:8
```

The request can include `run_id`, `prompt_id`, `task_type`, `arm`, `repeat`,
`model`, and `ragas`. `run_id` is indexed uniquely, so retrying the same run
ID is rejected instead of creating a duplicate measurement.

The driver lifecycle and async client usage follow the current MongoDB Python
driver documentation:
https://www.mongodb.com/docs/languages/python/pymongo-driver/current/connect/mongoclient/

## Ragas evaluation

Each completed run is evaluated in the background for its selected `arm`. The
MongoDB document is first written with `ragas.status: "pending"`, then updated
with `faithfulness`, `relevance`, and `answer_relevancy` scores. Faithfulness
is evaluated when the run has retrieved context; direct arms still receive
answer relevancy evaluation.

The evaluator uses Ragas's modern `Faithfulness` and `AnswerRelevancy`
collection metrics with the configured Google GenAI model. Evaluation errors
are stored as `ragas.status: "failed"` and do not delay the frontend response.
