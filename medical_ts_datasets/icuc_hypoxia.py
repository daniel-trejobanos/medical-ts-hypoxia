"""icuc_hypoxia dataset."""
from os.path import join
from collections.abc import Sequence
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pyarrow.parquet as pq
import pandas as pd

from medical_ts_datasets.util import MedicalTsDatasetBuilder, MedicalTsDatasetInfo


_CITATION = """
"""

_DESCRIPTION = """
"""



class ICUCHReader(Sequence):
    """
    Reader class used to load a data set to the medical ts interface
    """
    # TODO: read this from config file instead

    vital_features = ['MAP', 'ICP_1', 'ICP_2', 'PbtO2_1',
       'PbtO2_2', 'CPP_1', 'CPP_2']
    ts_features = vital_features

    def __init__(self, data_path, listfile):
        """
        Class constructor

        Args:
           self arg1
           data_dir directory where cases subdirectories are
           listfile file with list of cases in data subset, csv with column CaseID

        """

        self.data_paths = data_path
        #here you read the file catalog
        self.samples = pd.read_csv(listfile)
        self.label_dtype = np.float32
        self.data_dtype = np.float32

    def __getitem__(self, index):
        """
        Retrieve case target and time series

        Args:
           index index of case to read

        Returns:
            case_id Id number for the case retrieved
            dict    Dictionary with time stamps, time series and target


        """
        case_id = str(self.samples.iloc[index,self.samples.columns.get_loc('CaseID')])
        data_file = join(self.data_paths, case_id, "processed_case.parquet")

        data = pq.read_table(data_file).to_pandas()

        # We convert from ns from admission to s from admission
        time =  (data.index.values- data.index.values[0])/np.timedelta64(1,"s").astype(np.int32)

        vitals = data[self.vital_features]

        target_sensors = {"critical_events_PbtO2_1", "critical_events_PbtO2_2"}
        if 'critical_events_PbtO2_1' in data.columns:
            hypoxia_label_1 = data['critical_events_PbtO2_1']
        else:
            hypoxia_label_1 = False
        if 'critical_events_PbtO2_2' in data.columns:
            hypoxia_label_2 = data['critical_events_PbtO2_2']
        else:
            hypoxia_label_2 = False

        hypoxia_label = hypoxia_label_1 | hypoxia_label_2
        not_null_events = hypoxia_label.notnull()

        time = time[not_null_events]
        vitals= vitals[not_null_events]
        hypoxia_label = hypoxia_label[not_null_events]

        return case_id, {
            # 'demographics': None,
            'time': time.astype(self.data_dtype),
            'vitals': vitals.values.astype(self.data_dtype),
            # 'lab_measurements': None,
            'targets': {
                'critical_events': hypoxia_label.values.astype(np.int32)[:, np.newaxis]
            },
            'metadata': {
                'patient_id': case_id
            }
        }

    def __len__(self):
        """Return number of instances in the dataset."""
        return len(self.samples)


class ICUCHypoxia(MedicalTsDatasetBuilder):

    VERSION = tfds.core.Version('1.0.1')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Register into https://example.org/login to get the data. Place the `data.zip`
    file in the `manual_dir/`.
    """
    has_demographics = False
    has_vitals = True
    has_lab_measurements = False
    has_interventions = False
    default_target = 'critical_events'

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
            targets={
               'critical_events':
                    tfds.features.Tensor(
                        shape=(None, 1), dtype=tf.int32)
            },
            default_target='critical_events',
            vitals_names=ICUCHReader.vital_features,
            description=_DESCRIPTION,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        # TODO: this has to be done from the builder call
        # dl_config = tfds.download.DownloadConfig(manual_dir=Path())

        archive_path = dl_manager.manual_dir


        train_dir = archive_path
        train_listfile = join(archive_path , 'train_listfile.csv')
        val_dir = archive_path
        val_listfile = join(archive_path , 'val_listfile.csv')
        test_dir =  archive_path
        test_listfile = join(archive_path , 'test_listfile.csv')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_dir': train_dir,
                    'listfile': train_listfile,
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'data_dir': val_dir,
                    'listfile': val_listfile,
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_dir': test_dir,
                    'listfile': test_listfile,
                }
            ),
        ]

    def _generate_examples(self, data_dir, listfile):
        """Yield examples."""
        reader = ICUCHReader(data_dir, listfile)
        for case_id, instance in reader:
            yield case_id, instance
