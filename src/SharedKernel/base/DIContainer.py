import importlib
import inspect
import pkgutil
from typing import Set
from lagom import Container as LagomContainer

class DIContainer(LagomContainer):
    def __init__(self, base_package: str = None):
        super().__init__()
        self.base_package = base_package or "src"
        self._registered_classes = set() 
        self._registered_keys = set() 
        self._scan_and_register()

    def _scan_and_register(self):
        print(f"[DI] Scanning package: {self.base_package} (Mode: Concrete + Interface)")

        try:
            package = importlib.import_module(self.base_package)
        except ImportError:
            print(f"[DI] Error: Package '{self.base_package}' not found.")
            return

        for _, module_name, is_pkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            # Bỏ qua sub-packages để tránh scan quá sâu (tùy chọn)
            if is_pkg: 
                continue 
            
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"[DI] Skip module {module_name} due to import error: {e}")
                continue

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if not hasattr(obj, "__di_type__"):
                    continue

                if obj in self._registered_classes:
                    continue

                target_key = None

                # Explicit interface từ decorator
                if hasattr(obj, "__di_interface__"):
                    target_key = obj.__di_interface__
                    # print(f"[DI] -> [{name}] implements explicit interface: {target_key.__name__}")
                
                if not target_key:
                    target_key = obj
                    # print(f"[DI] -> [{name}] has NO interface. Registering as Concrete Type.")
                
                if target_key in self._registered_keys:
                    # print(f"[DI] ---- Skipping duplicate registration for key: {getattr(target_key, '__name__', str(target_key))}")
                    continue

                # Binding: Key -> Implementation Class
                self[target_key] = obj
                
                self._registered_classes.add(obj)
                self._registered_keys.add(target_key)

                # key_name = getattr(target_key, '__name__', str(target_key))
                # print(f"[DI] ---- Registered [{key_name}] -> [{name}]")
        print("[DI] Scan completed!")
    ...