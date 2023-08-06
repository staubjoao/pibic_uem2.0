import dcase_util

# Setup logging
dcase_util.utils.setup_logging()

log = dcase_util.ui.FancyLogger()
log.title('Acoustic Scene Classification Example / GMM')

# Create dataset object and set dataset to be stored under 'data' directory.
db = dcase_util.datasets.TUTAcousticScenes_2016_EvaluationSet(
    data_path='evaluationset_2016'
)

# Initialize dataset (download, extract and prepare it).
db.initialize()

# Show dataset information
db.show()
