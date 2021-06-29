from app.main import app
import pickle as p

if __name__ == "__main__":
    heartmodelfile = 'pkl/heart_model.pickle'
    heartModel = p.load(open(heartmodelfile, 'rb'))
    app.run()
