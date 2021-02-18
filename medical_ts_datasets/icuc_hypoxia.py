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

    blacklist = None
    parameters = ['MAP', 'ICP_1', 'ICP_2', 'PbtO2_1',
       'PbtO2_2', 'CPP_1', 'CPP_2']

    def __init__(self, data_dir, listfile):
        """
        Class constructor

        Args:
           self arg1
           data_dir directory where cases subdirectories are
           listfile file with list of cases in data subset, csv with column CaseID

        """

        self.dataset_dir = data_dir
        #here you read the file catalog
        self.instances = pd.read_csv(listfile)

        if self.blacklist is not None:
            # remove instances which are on the blacklist
            self.instances = self.instances[
                ~self.instances['Caseid'].isin(blacklist)
            ]


    def __getitem__(self, index):
        """
        Retrieve case target and time series

        Args:
           index index of case to read

        Returns:
            case_id Id number for the case retrieved
            dict    Dictionary with time stamps, time series and target


        """
        case_id = str(self.instances.iloc[index,self.instances.columns.get_loc('CaseID')])
        data_file = join(self.dataset_dir, case_id, "processed_case.parquet")

        data = pq.read_table(data_file).to_pandas()

        # We convert from ns from admission to s from admission
        time =  (data.index.values- data.index.values[0])/np.timedelta64(1,"s").astype(np.int32)
        breakpoint()

        parameters_readings = data[self.parameters]
        hypoxia_label = data['critical_events_PbtO2_2']

        not_null_events = hypoxia_label.notnull()

        time = time[not_null_events]
        parameters_readings= parameters_readings[not_null_events]
        hypoxia_label = hypoxia_label[not_null_events]

        return case_id, {
            'time': time.astype(np.float32),
            'vitals': parameters_readings.values.astype(np.float32),
            'targets': {
                'critical_events_PbtO2_2': hypoxia_label.values.astype(np.int32)[:, np.newaxis]
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
    default_target = 'critical_events_PbtO2_2'

    def _info(self):
        return MedicalTsDatasetInfo(
            builder=self,
            targets={
               'critical_events_PbtO2_2':
                    tfds.features.Tensor(
                        shape=(None, 1), dtype=tf.int32)
            },
            default_target='critical_events_PbtO2_2',
            vitals_names=ICUCHReader.parameters,
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