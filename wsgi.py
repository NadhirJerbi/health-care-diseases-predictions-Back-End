from app.main import app
import pickle as p

if __name__ == "__main__":
    heartmodelfile = 'pkl/final_prediction.pickle'
    heartModel = p.load(open(heartmodelfile, 'rb'))
    app.run()
