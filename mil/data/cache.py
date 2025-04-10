from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from typing import List, Tuple, Callable, Any

from tqdm.auto import tqdm


class CacheToRAM:
    def __init__(
            self,
            data: List[Tuple[str, str]],
            load_function: Callable[[str], Any],
            workers: int = 8,
            parallel: str = None,
    ):
        """
        Caches data in RAM using multithreading multiprocessing.

        Args:
            data (List[Tuple[str, str]]): List of pairs where each pair contains an identifier and a path.
            load_function (Callable[[str], Any]): Function to load data given a path.
            workers (int, optional): Number of worker processes for multiprocessing. Defaults to 8.
        """
        # default "threading"
        if parallel is None:
            self.parallel = "threading"
        else:
            self.parallel = parallel

        self.data = data
        self.load_function = load_function
        self.workers = workers
        self.__cache_dict = {}

    def _load_sample(self, sample: Tuple[str, str]) -> Tuple[str, Any]:
        """
        Wrapper function for multiprocessing to load data.

        Args:
            sample (Tuple[str, str]): Data pair with identifier and path.

        Returns:
            Tuple[str, Any]: Identifier and loaded data.
        """
        identifier, path = sample
        return identifier, self.load_function(path)

    def cache_data(self):
        """
        Cache data using multiprocessing.
        """
        self.__cache_dict = {}

        if self.parallel == "threading":
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self._load_sample, sample): sample for sample in self.data}

                for future in tqdm(as_completed(futures), total=len(self.data), desc='Loading data to RAM', leave=False):
                    identifier, data = future.result()
                    self.__cache_dict[identifier] = data

        elif self.parallel == "multiprocessing":
            with Pool(self.workers) as pool:
                results = pool.imap(self._load_sample, self.data)
                self.__cache_dict = dict(tqdm(results, total=len(self.data), desc='Loading data to RAM', leave=False))

        else:
            raise NotImplementedError(f"Parallel mode not implemented for {self.parallel}")

    @property
    def cache_dict(self):
        return self.__cache_dict