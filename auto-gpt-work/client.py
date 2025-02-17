from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

@app.post("/query")
def get_performance(query: QueryRequest):
    # 处理查询并返回结果
    result = perform_query(query.user_query)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
