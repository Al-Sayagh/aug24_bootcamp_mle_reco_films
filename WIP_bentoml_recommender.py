import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class UserInput(BaseModel):
    user_id: int

reco_runner = bentoml.sklearn.get("recommender_model:latest").to_runner()

svc = bentoml.Service(
    name="recommender_service",
    runners=[reco_runner],
)

@svc.api(input=JSON(pydantic_model=UserInput), output=JSON())
def recommend(input_data: UserInput):
    # Logique de recommandation
    predictions = reco_runner.run(input_data.user_id)
    return {"recommendations": predictions}
