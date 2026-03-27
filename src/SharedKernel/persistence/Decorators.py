
import asyncio
from functools import wraps
import functools
from typing import Any, Callable, Optional, Type
import traceback

def Transactional(cls):
    """
    Decorator để tự wrap tất cả các method async
    của class với decorator `transaction`.
    """
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and asyncio.iscoroutinefunction(attr_value):
            setattr(cls, attr_name, Transaction(attr_value))
    return cls

def Transaction(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        try:
            result = await func(self, *args, **kwargs)
            # await self.session.commit()
            print(f"Transaction committed - Session ID: {id(self.session)}")
            return result
        except Exception as e:
            await self.session.rollback()
            print(f"Transaction rolled back - Session ID: {id(self.session)}, Error: {e}")
            raise
        # finally:
        #     await self.session.close()
        #     print(f"Session closed - Session ID: {id(self.session)}")
    return wrapper

def Service(implements: Optional[Type] = None):
    def decorator(cls):
        cls.__di_type__ = "service"
        
        if implements:
            cls.__di_interface__ = implements
        
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def validated_init(self, *args, **kwargs):
            try:
                return original_init(self, *args, **kwargs)
            except Exception as e:
                print(f"[DI Error] Initialization failed for {cls.__name__}: {e}")
                raise
        cls.__init__ = validated_init
        return cls
    
    # Xử lý cú pháp @Service() hoặc @Service
    if callable(implements):
        cls = implements
        return decorator(cls)
    
    return decorator

def Repository(implements: Optional[Type] = None):
    def decorator(cls):
        cls.__di_type__ = "repository"
        
        # Lưu trữ interface target nếu có
        if implements:
            cls.__di_interface__ = implements
        
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def validated_init(self, *args, **kwargs):
            try:
                return original_init(self, *args, **kwargs)
            except Exception as e:
                # Log lỗi khởi tạo chi tiết
                print(f"[DI Error] Initialization failed for {cls.__name__}: {e}")
                raise
        cls.__init__ = validated_init
        return cls
    
    # Xử lý cú pháp @Service() hoặc @Service
    if callable(implements):
        cls = implements
        return decorator(cls)
    
    return decorator

def Controller(cls: Type[Any]) -> Type[Any]:
    cls.__di_type__ = "controller"

    original_init = cls.__init__
    
    def validated_init(self, *args, **kwargs):
        try:
            return original_init(self, *args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            # traceback.print_exc()
            raise
    
    cls.__init__ = validated_init
    return cls