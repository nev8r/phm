"""
Cache utility module

Purpose: provide utility helpers used by the bearing PHM framework
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import pickle
from pathlib import Path

# import dill

from USTC.SSE.BearingPrediction.util.Logger import Logger


class Cache:
    __CACHE_DIR = Path('cache')
    __CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __init__(self, cache_dir: str):
        Cache.__CACHE_DIR = Path(cache_dir)
        Cache.__CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def __get_cache_file(cls, name: str) -> Path:
        """
        根据信息获取缓存文件（名称及位置）
        :param name:
        :return:
        """
        # hash_input = str(kwargs).encode('utf-8')
        # hash_value = hashlib.md5(hash_input).hexdigest()
        return cls.__CACHE_DIR / f'{name}.pkl'

    @classmethod
    def save(cls, target, name):
        """
        保存缓存到文件
        :return:
        """
        cache_file = cls.__get_cache_file(name)
        with cache_file.open('wb') as f:
            Logger.debug(f"[Cache]  Generating cache file: {cache_file}")
            pickle.dump(target, f)
            # dill.dump(target, f)
        Logger.debug(f"[Cache]  Generated cache file: {cache_file}")

    @classmethod
    def load(cls, name, is_able=True):
        """
        从文件加载缓存
        :return:
        """
        if not is_able:
            return None

        cache_file = cls.__get_cache_file(name)
        if cache_file.exists():
            with cache_file.open('rb') as f:
                Logger.debug(f"[Cache]  -> Loading cache file: {cache_file}")
                cache = pickle.load(f)
                # cache = dill.load(f)
                Logger.debug(f"[Cache]  Successfully loaded: {cache_file}")
                return cache
        else:
            Logger.info(f'[Cache]  Cache miss: {cache_file}')
            return None

    @classmethod
    def delete(cls, name, is_able=True):
        """
        删除缓存文件
        :param name:
        :param is_able:
        :return:
        """
        if not is_able:
            return None

        cache_file = cls.__get_cache_file(name)
        if cache_file.exists():
            cache_file.unlink()
            Logger.debug(f"[Cache]  Deleted cache file {cache_file}")
        else:
            Logger.info(f"[Cache]  Delete skipped, cache file is absent: {cache_file}")
