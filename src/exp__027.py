"""usage: exp__027.py [-h] [--input INPUT] [--output OUTPUT] [--force] [--simple]

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    input directory (default: /analysis/data/inputs)
  --output OUTPUT  output directory (default: /analysis/data/outputs)
  --force          If add me, re-create all models. (default: False)
  --simple         if add me, create lightgbm model only. (skip other models)
                   (default: False)
"""

import colorsys
import os
import random
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass
from typing import List
from typing import Union

import joblib
import numpy as np
import pandas as pd
import texthero as hero
import torch
from PIL import ImageColor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tabulate import tabulate
from vivid import create_runner
from vivid.backends import LocalExperimentBackend
from vivid.cacheable import cacheable
from vivid.env import Settings
from vivid.estimators.base import MetaBlock
from vivid.estimators.boosting import LGBMRegressorBlock
from vivid.estimators.boosting import XGBRegressorBlock
from vivid.estimators.boosting.block import create_boosting_seed_blocks
from vivid.estimators.ensumble import RFRegressorBlock
from vivid.estimators.linear import TunedRidgeBlock
from vivid.features.base import CountEncodingBlock, OneHotEncodingBlock, FilterBlock
from vivid.core import BaseBlock as VividBaseBlock
from vivid.setup import setup_project
from vivid.utils import Timer
import fasttext
import fasttext.util


class FillnaBlock(VividBaseBlock):
    _save_attributes = [
        'fill_values_'
    ]

    def fit(self,
            source_df: pd.DataFrame,
            y: Union[None, np.ndarray],
            experiment) -> pd.DataFrame:
        self.fill_values_ = source_df.median()
        return self.transform(source_df)

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        out_df = source_df.replace([np.inf, -np.inf], np.nan).fillna(self.fill_values_).fillna(0)
        return out_df


class BaggingSVRegressorBlock(MetaBlock):
    def model_class(self, *args, **kwargs):
        return BaggingRegressor(
            base_estimator=make_pipeline(
                StandardScaler(),
                SVR(*args, **kwargs)
            ),
            n_estimators=10,
            max_samples=.2,
            n_jobs=-1
        )


OBJECT_ID = 'object_id'


def seed_everything(seed=510):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


@cacheable
def read_csv(name: str) -> pd.DataFrame:
    if '.csv' not in name:
        name = name + '.csv'

    project = setup_project()
    return pd.read_csv(os.path.join(project.output_root, name))


@cacheable
def read_whole_df() -> pd.DataFrame:
    return pd.concat([
        read_csv('train'),
        read_csv('test')
    ], ignore_index=True)


def preprocess(input_df):
    out_df = input_df.copy()

    date = pd.to_datetime(input_df['acquisition_date'])

    out_df['acquisition_date'] = date
    out_df['acquisition_date_year'] = date.dt.year
    out_df['acquisition_date_month'] = date.dt.month
    return out_df


def crop_text(s, max_length=30):
    if len(s) < max_length:
        return s

    return s[:max_length] + '…'


class BaseBlock(object):
    def fit(self,
            input_df: pd.DataFrame,
            y=None, **kwrgs) -> pd.DataFrame:
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class StringLengthBlock(BaseBlock):
    def __init__(self, name: str, columns, *args, **kwargs):
        self.columns = columns

    def transform(self, source_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        for c in self.columns:
            x = source_df[c].fillna('')
            out_df[c] = x.str.len()

        return out_df


def merge_by_key(left: Union[pd.DataFrame, pd.Series],
                 right: pd.DataFrame,
                 on=OBJECT_ID) -> pd.DataFrame:
    if not isinstance(left, pd.Series):
        left = left[on]
    return pd.merge(left, right, on=on, how='left').drop(columns=[on])


class One2ManyBlock(BaseBlock):
    def __init__(self, name, many_df):
        self.many_df = many_df
        self.name = name

    def fit(self, input_df, y=None, **kwrgs):
        minimum_freq = 30
        vc = self.many_df['name'].value_counts()
        use_values = vc[vc > minimum_freq].index
        use_df = self.many_df[self.many_df['name'].isin(use_values)].reset_index(drop=True)

        self.agg_df_ = pd.crosstab(use_df['object_id'], use_df['name']).reset_index()
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df_).fillna(0).astype(int)  # type: pd.DataFrame
        out_df['total_occurance'] = out_df.sum(axis=1)
        out_df.columns = out_df.columns.map(crop_text)
        return out_df.add_prefix(self.name + '__')


class RelationCount(BaseBlock):
    def __init__(self, name):
        self.name = name

    def fit(self, input_df, y=None, **kwrgs):
        other_df = read_csv(self.name)

        self.agg_df_ = other_df.groupby(OBJECT_ID).size().rename('size').reset_index()
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df_).fillna(0).astype(int)
        return out_df.add_prefix(f'Counts_{self.name}_')


def parse_year(s: str):
    """maker data の date of birth / death を parse する method"""
    if s is None:
        return None

    if isinstance(s, float):
        return s

    if '-' not in s:
        return int(s)

    return int(s.split('-')[0])


class MakerYearBlock(BaseBlock):

    def fit(self, input_df, y=None, **kwrgs):
        maker_df = read_csv('maker')

        output_df = maker_df[['name']].copy()
        output_df['birth_year'] = maker_df['date_of_birth'].map(parse_year)
        output_df['death_year'] = maker_df['date_of_death'].map(parse_year)
        output_df['living_year'] = output_df['death_year'] - output_df['birth_year']
        output_df['birth_and_death_same_place'] = (maker_df['place_of_birth'] == maker_df['place_of_death']).astype(int)
        self.agg_df_ = output_df
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df['principal_maker'].rename('name'), self.agg_df_, on='name')
        out_df['diff_death_and_dating'] = input_df['dating_year_late'] - out_df['death_year']
        return out_df.add_prefix('principal_maker_info_')


def create_color_rgb():
    color_df = read_csv('color')
    main_color_df = color_df.sort_values('percentage', ascending=False).groupby('object_id').first().reset_index()
    out_df = pd.DataFrame(main_color_df['hex'].str.strip().map(ImageColor.getrgb).values.tolist(),
                          columns=['R', 'G', 'B'])
    return out_df, main_color_df


class MainColorBlock(BaseBlock):
    def fit(self, input_df, y=None, **kwrgs):
        rgb_df, main_color_df = create_color_rgb()
        self.agg_df_ = pd.concat([
            main_color_df[['object_id', 'percentage']],
            rgb_df
        ], axis=1)
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df_)
        return out_df.add_prefix('main_color_')


class PaletteBlock(BaseBlock):
    def transform(self, input_df):
        palette_df = read_csv('palette')
        agg_df = palette_df.groupby('object_id').first()
        out_df = merge_by_key(input_df, agg_df)
        return out_df.add_prefix('palette_')


def _extract_values(s):
    """`sub_title` の値から どの方向(key) / 値の数値 (value) / 単位 (atom) を取り出す."""
    try:
        key, value = s.strip().split(' ')
    except Exception:
        print(f'{s} is None')
        return None
    try:
        atom = re.findall("[a-zA-Z]+", value)[0]
    except Exception as e:
        atom = None

    if atom is not None:
        value = value.replace(atom, '')
    try:
        value = float(value)
    except Exception as e:
        print(value, s)
        value = np.nan

    return key, value, atom


@cacheable
def read_art_sub_title_parsed_data() -> pd.DataFrame:
    whole_df = read_whole_df()
    x = whole_df.set_index(OBJECT_ID)['sub_title'].str.split('×').explode().map(_extract_values)
    out_df = pd.DataFrame(x.values.tolist(), index=x.index, columns=['key', 'value', 'atom']).reset_index()
    return out_df


class ArtAttributeBlock(BaseBlock):
    def fit(self, input_df, y=None, experiment=None):
        atom_df = read_art_sub_title_parsed_data()
        agg_df = pd.pivot_table(data=atom_df,
                                index='object_id',
                                columns=['key'],
                                values='value',
                                aggfunc='mean')
        agg_df['area'] = agg_df['w'] * agg_df['h']

        # aspect比
        agg_df['aspect'] = agg_df['w'] / agg_df['h']
        agg_df['has_attributes'] = agg_df.isnull().sum(axis=1).fillna(0).astype(int)
        self.agg_df = agg_df
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df)
        return out_df.add_prefix('subtitle_attr_')


def text_normalization(text):
    import nltk
    # 英語とオランダ語を stopword として指定
    custom_stopwords = nltk.corpus.stopwords.words('dutch') + nltk.corpus.stopwords.words('english')

    x = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ])

    return x


class TfidfBlock(BaseBlock):
    def __init__(self, column, n_components=50):
        self.column = column
        self.n_components = n_components

    def __str__(self):
        return 'Tfidf_{}_{}'.format(self.column, self.n_components)
    
    def get_text_series(self, input_df):
        out_series = None
        
        for i, c in enumerate(self.column.split('&')):
            text_i = text_normalization(input_df[c]).astype(str)
            if out_series is None:
                out_series = text_i
            else:
                out_series = out_series + ' ' + text_i
            
        print(out_series)
        return out_series

    def fit(self, input_df, y=None, experiment=None):
        whole_df = read_whole_df()
        x = self.get_text_series(whole_df)

        transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=20000)),
            ('svd', TruncatedSVD(n_components=self.n_components, random_state=510))
        ])
        feature = transformer.fit_transform(x)
        self.agg_df = pd.concat([
            whole_df[[OBJECT_ID]].copy(),
            pd.DataFrame(feature)
        ], axis=1)
        return self.transform(input_df)

    def transform(self, input_df):
        return merge_by_key(input_df, self.agg_df).add_prefix(f'{self.column}_tfidf_')


def _create_rank(whole_df, material_df, name):
    idx = material_df['name'] == name
    objects = set(material_df[idx]['object_id'].unique())
    _df = whole_df.set_index(OBJECT_ID)
    idx = _df.index.isin(objects)
    _rank = _df[idx]['dating_sorting_date'].rank(method='min').rename(name)
    _df = merge_by_key(whole_df, _rank)
    return _df


class DatingRankByMaterialBlock(BaseBlock):
    def fit(self, input_df, y=None, experiment=None):
        material_df = read_csv('material')
        whole_df = read_whole_df()

        vc = material_df['name'].value_counts()
        names = vc[vc > 30].index
        df = whole_df[[OBJECT_ID]].copy()
        for n in names:
            df = pd.concat([df,
                            _create_rank(whole_df=whole_df, material_df=material_df, name=n)], axis=1)

        self.agg_df = df
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df)
        return out_df.add_prefix('Material_Date_rank_')


class PaintingDurationBlock(BaseBlock):
    def transform(self, input_df):
        out_df = pd.DataFrame({
            'dating_duration': input_df['dating_year_early'] - input_df['dating_year_late']
        })
        return out_df


class OtherMainColorBlock(BaseBlock):
    def fit(self, input_df, y=None, experiment=None):
        rgb_df, main_color_df = create_color_rgb()

        methods = [
            colorsys.rgb_to_hsv,
            colorsys.rgb_to_hls,
            colorsys.rgb_to_yiq
        ]

        def create_converted_color_df(func, rgb_df):
            return pd.DataFrame(
                [func(*x) for x in rgb_df.values],
            ).add_prefix(str(func.__name__))

        self.agg_df = pd.concat([
            main_color_df[OBJECT_ID],
            *[create_converted_color_df(func, rgb_df=rgb_df) for func in methods]
        ], axis=1)
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df)
        return out_df.add_prefix('OtherMainColor_')



def create_pivot_feature(input_df: pd.DataFrame, index: str, column: str, anterior=0, normalize=False) -> pd.DataFrame:
    """
    count base の pivot table を作成.

    Args:
        input_df:
            変換対象の dataframe
        index:
            index 名
        column:
            カラム名
        anterior:
            事前情報 (最低でもこの回数だけデータが出現したとみなす).
            normalize=False のときは無視される.
        normalize: bool
            True のとき出現回数の合計値で正規化を行う. 
            イメージで言うと index の各要素ごとの column のでやすさ, のような意味合いになる

    Returns:
        shape = (n_index_unique, n_column_unique) の dataframe

    """
    agg_df = pd.pivot_table(data=input_df, index=index, columns=column, aggfunc='size')
    agg_df = agg_df.fillna(0).astype(int)
    if normalize:
        return agg_df.add_prefix(f'{column}=')
   
    _z = (agg_df + anterior)
    normed = _z.div(_z.sum(axis=1), axis=0)
    return normed.add_prefix(f'{column}=')


class PrincipalMakerCountBlock(BaseBlock):
    def fit(self, input_df, y=None, experiment=None):
        agg_df = pd.DataFrame()
        principal_maker_df = read_csv('principal_maker')

        for c in ['qualification', 'roles', 'productionPlaces']:
            _df = pd.pivot_table(data=principal_maker_df,
                                 index='object_id', columns=c,
                                 aggfunc='size').fillna(0).astype(int)
            agg_df = pd.concat([
                agg_df,
                _df.add_prefix(f'{c}=')
            ], axis=1)

        self.agg_df = agg_df
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df).fillna(0).astype(int)
        return out_df.add_prefix('PrincipalMaker_')


class PrincipalMakerMetaBlock(BaseBlock):
    def fit(self, input_df, y=None, experiment=None):
        agg_df = pd.DataFrame()
        principal_maker_df = read_csv('principal_maker')

        for c in ['qualification', 'roles', 'productionPlaces']:
            _df = create_pivot_feature(
                input_df=principal_maker_df, 
                index='object_id', column=c,
                anterior=1, normalize=True)
                                       
            agg_df = pd.concat([
                agg_df,
                _df
            ], axis=1)

        self.agg_df = agg_df
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df).fillna(0).astype(int)
        return out_df.add_prefix('PrincipalMakerMeta_')

    
class PrincipalMakerOtherFieldMetaBlock(BaseBlock):
    def fit(self, input_df, y=None, **kwrgs):
        whole_df = read_whole_df().drop(columns=['likes'])
        group = whole_df.groupby('principal_maker')
        self.agg_df_ = pd.concat([
            group.nunique().add_prefix('nunique_'),
            (group.nunique() / group.size().values.reshape(-1, 1)).add_prefix('ratio_'),  
        ], axis=1)
               
        return self.transform(input_df)
    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df_, 'principal_maker')
        return out_df.add_prefix('PrincipalMakerOtherField_')


class FilledMakerBirthBlock(BaseBlock):
    def fit(self, input_df, y=None, experiment=None):
        whole_df = read_whole_df()
        maker_df = read_csv('maker')

        soring_year = whole_df.groupby('principal_maker')['dating_sorting_date'].min()
        soring_year = maker_df['name'].map(soring_year)
        agg_df = maker_df[['name']].copy()

        year = maker_df['date_of_birth'].map(parse_year)
        agg_df['filled_birth_date'] = np.where(year.isnull(), soring_year, year)
        self.agg_df = agg_df
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = merge_by_key(input_df, self.agg_df.rename(columns={'name': 'principal_maker'}), 'principal_maker')
        return out_df.add_prefix('FilledMakerInfo_')


class SameMakerBlock(BaseBlock):
    def transform(self, input_df):
        out_df = pd.DataFrame({
            'same_maker': (input_df['principal_maker'] != input_df['principal_or_first_maker']).astype(int)
        })
        return out_df


def convert_hex2rgb(hex_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(hex_series.str.strip().map(ImageColor.getrgb).values.tolist(),
                        columns=['R', 'G', 'B'])


class ConvertedMainColorBlock(BaseBlock):
    def fit(self, input_df, y=None, **kwrgs):
        color_df = read_csv('color')

        functions = [
            colorsys.rgb_to_hsv,
            colorsys.rgb_to_hls,
            colorsys.rgb_to_yiq
        ]

        agg_df = pd.DataFrame()

        for func in functions:
            values = [func(*x) for x in convert_hex2rgb(color_df['hex']).values]
            _df = pd.DataFrame(values) * color_df['percentage'].values.reshape(-1, 1)
            _df = _df.groupby(color_df['object_id']).mean().add_prefix(func.__name__)
            agg_df = pd.concat([agg_df, _df], axis=1)
        self.agg_df = agg_df
        return self.transform(input_df)

    def transform(self, input_df):
        return merge_by_key(input_df, self.agg_df).add_prefix('ConvertedMainColor_')


class TargetEncodingBlock(BaseBlock):
    def __init__(self, use_columns: List[str], cv):
        super(TargetEncodingBlock, self).__init__()

        self.mapping_df_ = None
        self.use_columns = use_columns
        self.cv = list(cv)
        self.n_fold = len(cv)

    def create_mapping(self, input_df, y):
        self.mapping_df_ = {}
        self.y_mean_ = np.mean(y)

        out_df = pd.DataFrame()
        target = pd.Series(y)

        for col_name in self.use_columns:
            keys = input_df[col_name].unique()
            X = input_df[col_name]

            oof = np.zeros_like(X, dtype=np.float)

            for idx_train, idx_valid in self.cv:
                _df = target[idx_train].groupby(X[idx_train]).mean()
                _df = _df.reindex(keys)
                _df = _df.fillna(_df.mean())
                oof[idx_valid] = input_df[col_name][idx_valid].map(_df.to_dict())

            out_df[col_name] = oof

            self.mapping_df_[col_name] = target.groupby(X).mean()

        return out_df

    def fit(self,
            input_df: pd.DataFrame,
            y=None, **kwrgs) -> pd.DataFrame:
        out_df = self.create_mapping(input_df, y=y)
        return out_df.add_prefix('TE_')

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()

        for c in self.use_columns:
            out_df[c] = input_df[c].map(self.mapping_df_[c]).fillna(self.y_mean_)

        return out_df.add_prefix('TE_')


class MakerMetaFeature(BaseBlock):
    def fit(self,
            input_df: pd.DataFrame,
            y=None, **kwrgs) -> pd.DataFrame:
        df = read_csv('principal_maker')
        occupation_df = read_csv('principal_maker_occupation')

        agg_df = pd.DataFrame(index=df['maker_name'].unique())

        for c in ['roles', 'qualification', 'productionPlaces']:
            _df = create_pivot_feature(input_df=df, index='maker_name', column=c, anterior=1, normalize=True)
            agg_df = pd.concat([agg_df, _df], axis=1)

        agg_df = pd.concat([
            agg_df,
            create_pivot_feature(
                input_df=pd.merge(occupation_df, df, on='id', how='left').rename(columns={'name': 'occupation'}),
                index='maker_name',
                column='occupation',
                normalize=True,
                anterior=1)
        ], axis=1)
        agg_df.index = agg_df.index.rename('principal_maker')
        self.agg_df_ = agg_df

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = merge_by_key(input_df, self.agg_df_, on='principal_maker')
        out_df = out_df.fillna(self.agg_df_.mean())
        return out_df.add_prefix('MakerMeta_')

    
class MakerCountBlock(BaseBlock):
    def fit(self,
            input_df: pd.DataFrame,
            y=None, **kwrgs) -> pd.DataFrame:
        df = read_csv('principal_maker')
        occupation_df = read_csv('principal_maker_occupation')

        agg_df = pd.DataFrame(index=df['maker_name'].unique())

        for c in ['roles', 'qualification', 'productionPlaces']:
            _df = create_pivot_feature(input_df=df, index='maker_name', column=c)
            agg_df = pd.concat([agg_df, _df], axis=1)

        agg_df = pd.concat([
            agg_df,
            create_pivot_feature(
                input_df=pd.merge(occupation_df, df, on='id', how='left').rename(columns={'name': 'occupation'}),
                index='maker_name',
                column='occupation')
        ], axis=1)
        agg_df.index = agg_df.index.rename('principal_maker')
        self.agg_df_ = agg_df

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = merge_by_key(input_df, self.agg_df_, on='principal_maker')
        out_df = out_df.fillna(self.agg_df_.mean())
        return out_df.add_prefix('MakerCount_')


class MakerOtherTableAggBlock(BaseBlock):
    def __init__(self, name):
        self.name = name

    def fit(self,
            input_df: pd.DataFrame,
            y=None, **kwrgs) -> pd.DataFrame:
        whole_df = read_whole_df()
        minimum_freq = 5
        many_df = read_csv(self.name)

        _df = pd.merge(many_df,
                       whole_df[['object_id', 'principal_maker']], on='object_id', how='left')

        vc = _df['name'].value_counts()
        use_values = vc[vc > minimum_freq].index
        idx = _df['name'].isin(use_values)
        _df = _df[idx].reset_index(drop=True)

        self.agg_df_ = pd.concat([
            _df.pivot_table(index='principal_maker', columns='name', aggfunc='size').fillna(0).add_prefix('name='),
            _df.groupby('principal_maker')['name'].nunique().rename('name_nunique'),
            _df.groupby('principal_maker')['name'].size().rename('name_size')
        ], axis=1)

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = merge_by_key(input_df, self.agg_df_, on='principal_maker')
        return out_df.add_prefix(f'Maker@{self.name}_')


class AcquisitionMetaFeature(BaseBlock):
    def __init__(self,
                 column='acquisition_date',
                 value='principal_maker'):
        self.column = column
        self.value = value

    def fit(self,
            input_df: pd.DataFrame,
            y=None, **kwrgs) -> pd.DataFrame:
        whole_df = read_whole_df()
        values = whole_df[self.value]
        column = self.column
        index = whole_df[column]
        document = values.groupby(index).apply(list).apply(lambda x: ' '.join(map(str, x)))

        clf = Pipeline([
            ('count', CountVectorizer(max_features=50)),
        ])
        z = clf.fit_transform(document.values).toarray()
        self.agg_df_ = pd.merge(whole_df[[column, OBJECT_ID]],
                                pd.DataFrame(z, index=document.index), on=column, how='left').drop(columns=[column])
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = merge_by_key(input_df, self.agg_df_)
        return out_df.add_prefix(f'{self.column}_{self.value}_tfidf_')


class AcquisitionAndDatingBlock(BaseBlock):
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame()
        date = pd.to_datetime(input_df['acquisition_date'])
        out_df['year_from_dating_to_acquisition'] = date.dt.year - input_df['dating_sorting_date']
        return out_df

    
def get_text_series(input_df: pd.DataFrame, column: str, sep='&'):
    out_series = None
    for i, c in enumerate(column.split(sep)):
        text_i = text_normalization(input_df[c]).astype(str)
        if out_series is None:
            out_series = text_i
        else:
            out_series = out_series + ' ' + text_i
    return out_series

def create_embedding(document: str, model):
    words = document.split(' ')
    x = [model.get_word_vector(w) for w in words]
    x = np.max(x, axis=0)
    return x


def load_fasttext_model():
    with Timer(prefix='load FastText Pretrained'):
        # 予め data dir に pretrained fasttext の重みを download しておく必要がある
        # see: https://fasttext.cc/docs/en/crawl-vectors.html
        # fasttext.utils: https://github.com/facebookresearch/fastText/blob/master/python/fasttext_module/fasttext/util/util.py#L183
        ft = fasttext.load_model('/analysis/data/cc.nl.300.bin')
        ft = fasttext.util.reduce_model(ft, 100)
    return ft

class FasttextEmbeddingBlock(BaseBlock):    
    def __init__(self, column):
        self.column = column
        
    def fit(self, input_df, y, **kwrgs):
        ft = load_fasttext_model()
        whole_df = read_whole_df()
        text = get_text_series(whole_df, column=self.column)
        emb = text.map(lambda x: create_embedding(x, ft)).values
        emb = np.array(emb.tolist())
        
        self.agg_df = pd.concat([
            whole_df['object_id'],
            pd.DataFrame(emb)
        ], axis=1)
        
        return self.transform(input_df)
    
    def transform(self, input_df):
        return merge_by_key(input_df, self.agg_df).add_prefix(f'{self.column}_FastText')


class CustomOneHotBlock(OneHotEncodingBlock):
    def create_new_engine(self, column_name: str):
        return self.engine(min_freq=15)


def create_feature(train_df: pd.DataFrame, test_df: pd.DataFrame, y, blocks):
    train_feat_df = pd.DataFrame()
    test_feat_df = pd.DataFrame()

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    for b_i in blocks:
        with Timer(prefix=str(b_i)):
            try:
                out_i = b_i.fit(train_df, y=y, experiment=None)
            except Exception as e:
                print(f'Error on {b_i} fit. ')
                raise e from e

        print(f'shape: {out_i.shape}')
        train_feat_df = pd.concat([train_feat_df, out_i], axis=1)

    for b_i in blocks:
        with Timer(prefix=str(b_i) + '_test'):
            out_i = b_i.transform(test_df)

        test_feat_df = pd.concat([test_feat_df, out_i], axis=1)

    def postprocess(input_df: pd.DataFrame):
        return input_df.replace([np.inf, -np.inf], np.nan)

    return postprocess(train_feat_df), postprocess(test_feat_df)


def get_filename_without_extension(filename: str):
    return os.path.basename(filename).split('.')[0]


@dataclass
class RuntimeEnv:
    input_dir: str
    output_root: str
    force: bool = False
    simple: bool = False

    @property
    def output_dirpath(self) -> str:
        """実験結果を出力するディレクトリ"""
        file_name = get_filename_without_extension(__file__)
        return os.path.join(self.output_root, 'experiments', file_name)

    def initialize(self):
        Settings.PROJECT_ROOT = self.input_dir
        os.makedirs(self.output_dirpath, exist_ok=True)



def create_runtime_environment() -> RuntimeEnv:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='input directory', default='/analysis/data/inputs')
    parser.add_argument('--output', help='output directory', default='/analysis/data/outputs')
    parser.add_argument('--force', action='store_true', help='If add me, re-create all models.')
    parser.add_argument('--simple', action='store_true', help='if add me, create lightgbm model only. (skip other models)')
    args = parser.parse_args()
    runtime_env = RuntimeEnv(
        input_dir=args.input,
        output_root=args.output,
        force=args.force,
        simple=args.simple
    )
    runtime_env.initialize()
    return runtime_env


def create_model_blocks(cv, simple=False) -> List[VividBaseBlock]:
    fillna_block = FillnaBlock(name='FNA')

    init_params = {
        'cv': cv
    }

    single_models = [
        create_boosting_seed_blocks(
            feature_class=LGBMRegressorBlock, prefix='lgbm_',
            add_init_params={
                'subsample_freq': 3,
                "learning_rate": .05,
                "n_estimators": 100000,
                "reg_lambda": 0.0010412663539433173,
                "reg_alpha": 0.0012991200429910267,
                "colsample_bytree": 0.4133455910399615,
                "subsample": 0.9866975872244232,
                "max_depth": 8,
                "min_child_weight": 18.357712495890333,
                "num_leaves": int(.7 * 2 ** 8),
            },
            init_params=init_params),
    ]
    if simple:
        return single_models

    single_models += [
        # bagging する SVR
        BaggingSVRegressorBlock(name='bagging_svr', parent=fillna_block, **init_params),
        
        # seed averaging (xgboost)
        create_boosting_seed_blocks(feature_class=XGBRegressorBlock, prefix='xgb_', add_init_params={
            'n_estimators': 10000, 'colsample_bytree': .3, 'learning_rate': .05
        }, init_params=init_params, n_seeds=3),
        
        # seed averaging (lightgbm)
        create_boosting_seed_blocks(feature_class=LGBMRegressorBlock, prefix='lgbm_poisson_', add_init_params={
            'n_estimators': 10000, 'colsample_bytree': .2, 'learning_rate': .05, 'objective': 'poisson',
            'eval_metric': 'rmse',
        }, init_params=init_params),
        
        # random forest
        RFRegressorBlock('rf', parent=fillna_block, add_init_param={'n_jobs': -1}, **init_params),
        
        # ridge (linear model)
        TunedRidgeBlock('ridge', parent=fillna_block, n_trials=50, **init_params),
    ]

    stacked_models = [
        # stacking する lightgbm
        LGBMRegressorBlock('stacked_lgbm', parent=single_models, add_init_param={
            'n_estimators': 10000, 'colsample_bytree': 1, 'learning_rate': .05
        }, **init_params),
        
        # stacking + もとの特徴量を使う lightgbm
        LGBMRegressorBlock('stacked_and_raw_lgbm', parent=[*single_models, fillna_block], add_init_param={
            'n_estimators': 10000, 'colsample_bytree': .4, 'learning_rate': .05
        },
                           **init_params),
        
        # stacking ridge
        TunedRidgeBlock('stacked_ridge', parent=single_models, n_trials=50, **init_params),
    ]

    return stacked_models


def decoration(s, deco=None):
    if deco is None:
        deco = '=' * 30
    s = deco + s + deco
    return s


def run_pseudo_round(train_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     y,
                     cv,
                     output_dir,
                     n_train,
                     force: bool,
                     simple: bool) -> np.ndarray:
    # round が進んでいるときには学習用データに pseudo label の data を付与
    if len(train_df) != n_train:
        extend_index = range(n_train, len(train_df))
        cv = [[np.hstack([idx_tr, extend_index]), idx_val] for idx_tr, idx_val in cv]

    blocks = create_model_blocks(cv=cv, simple=simple)
    runner = create_runner(blocks, experiment=LocalExperimentBackend(output_dir))
    oof_results = runner.fit(train_df=train_df, y=y, ignore_past_log=force)

    y_origin = y[:n_train]

    scores = []
    for result in oof_results:
        score_i = mean_squared_error(y_origin, result.out_df.values[:n_train, 0]) ** .5
        scores.append([result.block.name, score_i])
    score_df = pd.DataFrame(scores, columns=['name', 'score']).sort_values('score').reset_index()
    print(decoration(' OOF Scores '))
    print(tabulate(score_df, headers='keys', tablefmt='psql'))
    score_df.to_csv(os.path.join(output_dir, 'score.csv'), index=False)

    test_results = runner.predict(test_df)

    best_test_pred = None
    base_oof_model_name = score_df['name'].values[0]

    for result in test_results:
        x = result.out_df.values[:, 0]
        x = np.expm1(x)
        x = np.where(x < 0, 0, x)

        if result.block.name == base_oof_model_name:
            print(f'use {result.block.name} for next predict')
            best_test_pred = np.log1p(np.copy(x))

        pd.DataFrame({
            'likes': x
        }).to_csv(os.path.join(output_dir, result.block.name + '.csv'), index=False)

    return best_test_pred


def main():
    runtime = create_runtime_environment()
    train_df = read_csv('train')
    test_df = read_csv('test')

    fold = KFold(n_splits=5, shuffle=True, random_state=71)
    # original の train と同じ長さで cv split を作る
    cv = list(fold.split(train_df))
    blocks = [
        PrincipalMakerOtherFieldMetaBlock(),
        AcquisitionAndDatingBlock(),
        MakerCountBlock(),
        *[One2ManyBlock(name, read_csv(name)) for name in
          ['material', 'technique', 'production_place', 'historical_person']],
        *[RelationCount(name) for name in
          ['material',
           'technique',
           'production_place',
           'historical_person',
           'color',
           'principal_maker']
          ],
        MakerYearBlock(),
        FilterBlock('filter', column=[
            'dating_period', 'dating_year_late', 'dating_year_early', 'dating_sorting_date'
        ]),
        CountEncodingBlock('CE', excludes=['likes']),
        CustomOneHotBlock('OH', column=[
            'title',
            'description',
            'long_title',
            'more_title',
            'sub_title',
            'acquisition_credit_line',
            'copyright_holder',
            'principal_maker',
            'acquisition_method',
            'acquisition_date_year',
            'acquisition_date'
        ]),
        StringLengthBlock('LEN', columns=[
            'title',
            'description',
            'long_title',
            'more_title',
            'sub_title',  # 不要と思いきやあったほうがスコアいい
            'acquisition_credit_line'
        ]),
        MainColorBlock(),
        *[TfidfBlock(column=column, n_components=50) for column in [
            'title',
            'description',
            'long_title',
            'more_title',
            'title&description&long_title&more_title',
            'acquisition_credit_line'
        ]],
        *[FasttextEmbeddingBlock(column=column) for column in [
            'title',
            'description',
            'long_title',
            'title&description&long_title&more_title',
        ]],
        TargetEncodingBlock(use_columns=[
            'title',
            'description',
            'long_title',
            'more_title',
            'sub_title',
            'acquisition_credit_line',
            'copyright_holder',
            'principal_maker',
            'acquisition_method',
            'acquisition_date_year',
            'acquisition_date'
        ], cv=cv),
        ArtAttributeBlock(),
        PaintingDurationBlock(),
        OtherMainColorBlock(),
        FilledMakerBirthBlock(),
        SameMakerBlock(),
        PrincipalMakerCountBlock(),
        # PrincipalMakerMetaBlock(),
        ConvertedMainColorBlock(),
        MakerMetaFeature(),
        *[MakerOtherTableAggBlock(n) for n in ['material', 'technique', 'production_place', 'historical_person']]
    ]

    y = np.log1p(train_df['likes'].values)
    feat_train_df, feat_test_df = create_feature(train_df, test_df, y=y, blocks=blocks)
    joblib.dump(feat_train_df, os.path.join(runtime.output_dirpath, 'train_feat.joblib'))
    joblib.dump(feat_test_df, os.path.join(runtime.output_dirpath, 'test_feat.joblib'))

    feat_df = feat_train_df.copy()
    run_pseudo_round(feat_df,
                     feat_test_df,
                     y,
                     cv=cv,
                     output_dir=runtime.output_dirpath,
                     n_train=len(train_df),
                     force=runtime.force,
                     simple=runtime.simple)


if __name__ == '__main__':
    main()
