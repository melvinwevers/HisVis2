
import argparse
from fastai.vision.all import *
from torch.serialization import default_restore_location
import torchvision.models as models
from sklearn.metrics import classification_report

from helper import *


def load_places():
    # th architecture to use
    arch = 'resnet50'

    # (down)load the pre-trained weights
    model_weights = '%s_places365.pth.tar' % arch
    if not os.access(model_weights, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_weights
        os.system('wget ' + weight_url)
        
    places_res50 = torch.load(model_weights,  
                            map_location=lambda storage, 
                            loc: storage)

    default_res50 = models.resnet50()
    state_dict = places_res50['state_dict'] 
    new_state_dict = OrderedDict()

    for key in state_dict.keys():
        new_state_dict[key[7:]]= state_dict[key]
        

    default_res50.fc = torch.nn.Linear(2048, 365) # Matching with default res50 dense layer
    default_res50.load_state_dict(new_state_dict)
    return default_res50

def export_classification_report(learn, output_path, current_time):
    interp = ClassificationInterpretation.from_learner(learn)
    interp.print_classification_report()

    predictions_,test_labels_ = flatten_check(interp.decoded, interp.targs)
    report = classification_report(test_labels_, predictions_, 
                                   labels=list(interp.vocab.o2i.values()), 
                                   target_names=[str(v) for v in interp.vocab],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    filename = current_time + '_classification_reportPlaces.csv'
    report_df.to_csv(os.path.join(output_path, filename))




def main(training_path, n_epochs, lr, output_path):

    current_time = time.strftime("%Y%m%d")

    # setting path variables 
    Path.BASE_PATH = Path(training_path)
    #Path.BASE_PATH.ls()
    path = Path.BASE_PATH
    print(path)	
    data = get_dls(128, 224, path)
    classes = data.vocab
    print(classes)
    default_res50 = load_places()

    learn = cnn_learner(get_dls(64, 224, path, augment=True),
                    new_resnet,
                    #models.resnet34,
                    metrics=[top_k_accuracy, 
                             accuracy,
                             error_rate])

    # train model
    learn.fit_one_cycle(n_epochs, lr, cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=5)])

    # evaluate model
    export_classification_report(learn, output_path, current_time)

    # save model
    filename = current_time + 'DeBoerPlaces.pkl'   
    learn.export(os.path.join(output_path, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, default='../../MelvinWevers#9512/DeBoer_Step1/')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--output_path', type=str, default='./output/models')
    args = parser.parse_args()

    if not os.path.exists('./output/models'):
        os.makedirs('./output/models')
    
    main(args.training_data_path, args.n_epochs, args.lr, args.output_path)
