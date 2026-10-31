from typing import List


class ParameterSubsetSelector:

    def __init__(self, pipe):
        self.pipe = pipe

    def find(self, startswith: List[str] = None, contains: List[str] = None, contains_substring: List[str] = None, exclude: List[str] = None, endswith: List[str] = None):
        print("Finding Parameter Subset for the given conditions")
        print("> startswith:", startswith)
        print("> contains:", contains)
        print("> contains_substring:", contains_substring)
        print("> endswith:", endswith)

        target_modules = {}
        for module_name, module in self.pipe.named_modules():

            module_name_tokens = module_name.split(".")

            # Skip the module if it's not in the specified startswith submodules
            if (startswith is not None) and not any(token == module_name_tokens[0] for token in startswith):
                continue

            # Skip the module if it does not contain all the specified tokens
            if (contains is not None) and not all(token in module_name_tokens for token in contains):
                continue

            # Skip the module if it does not contain all the specified tokens
            if (contains_substring is not None) and not all(substring in module_name for substring in contains_substring):
                continue

            # Skip the module if it contains any of the specified tokens
            if (exclude is not None) and any(token in module_name_tokens for token in exclude):
                continue

            # Skip the module if it does not end with any of the specified tokens
            if (endswith is not None) and not any(token == module_name_tokens[-1] for token in endswith):
                continue

            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                target_modules[module_name] = module

        return target_modules
