from fastapi import FastAPI 

app = FastAPI()

@app.get('/api/model1/{value}')
def model1(value: int):
    return {"results": value+1}

@app.get('/api/model2/{value}')
def model2(value: int):
    return {"results": value*2}
