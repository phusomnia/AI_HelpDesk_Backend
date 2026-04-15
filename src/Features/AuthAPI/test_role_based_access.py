"""
Unit tests cho RoleBasedAccess module.
Chạy tests: pytest test_role_based_access.py -v
"""

import pytest
from fastapi import Request
from unittest.mock import Mock, patch
from starlette.datastructures import Headers

from src.Features.AuthAPI.RoleBasedAccess import RoleBasedAccess, get_current_user, get_current_role, get_current_user_id
from src.Domain.base_entities import AccountsRole
from src.SharedKernel.exception.APIException import APIException


class TestRoleBasedAccess:
    """Test class cho RoleBasedAccess"""
    
    def setup_method(self):
        """Setup cho mỗi test"""
        self.role_access = RoleBasedAccess()
    
    def test_extract_token_from_header_valid(self):
        """Test 1: Extract token từ header hợp lệ"""
        # Arrange
        request = Mock(spec=Request)
        request.headers = Headers({"Authorization": "Bearer valid_token_123"})
        
        # Act
        token = self.role_access._extract_token_from_header(request)
        
        # Assert
        assert token == "valid_token_123"
    
    def test_extract_token_from_header_missing(self):
        """Test 2: Missing Authorization header"""
        # Arrange
        request = Mock(spec=Request)
        request.headers = Headers({})
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            self.role_access._extract_token_from_header(request)
        
        assert exc_info.value.status_code == 401
        assert "Missing Authorization header" in str(exc_info.value.message)
    
    def test_extract_token_from_header_invalid_format(self):
        """Test 3: Invalid Authorization header format"""
        # Arrange
        request = Mock(spec=Request)
        request.headers = Headers({"Authorization": "Basic dXNlcjpwYXNz"})
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            self.role_access._extract_token_from_header(request)
        
        assert exc_info.value.status_code == 401
        assert "Invalid Authorization header format" in str(exc_info.value.message)
    
    def test_extract_token_from_header_empty_token(self):
        """Test 4: Empty token sau Bearer"""
        # Arrange
        request = Mock(spec=Request)
        request.headers = Headers({"Authorization": "Bearer "})
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            self.role_access._extract_token_from_header(request)
        
        assert exc_info.value.status_code == 401
        assert "Empty token" in str(exc_info.value.message)
    
    @patch('src.Features.AuthAPI.RoleBasedAccess.JWTProvider')
    def test_verify_token_valid(self, mock_jwt_provider_class):
        """Test 5: Verify token hợp lệ"""
        # Arrange
        mock_jwt_provider = Mock()
        mock_jwt_provider.verify_token.return_value = {
            "username": "testuser",
            "role": "ADMIN",
            "user_id": "123"
        }
        mock_jwt_provider_class.return_value = mock_jwt_provider
        
        role_access = RoleBasedAccess(mock_jwt_provider)
        
        # Act
        payload = role_access._verify_token("valid_token")
        
        # Assert
        assert payload["username"] == "testuser"
        assert payload["role"] == "ADMIN"
        assert payload["user_id"] == "123"
    
    @patch('src.Features.AuthAPI.RoleBasedAccess.JWTProvider')
    def test_verify_token_invalid(self, mock_jwt_provider_class):
        """Test 6: Verify token không hợp lệ (None)"""
        # Arrange
        mock_jwt_provider = Mock()
        mock_jwt_provider.verify_token.return_value = None
        mock_jwt_provider_class.return_value = mock_jwt_provider
        
        role_access = RoleBasedAccess(mock_jwt_provider)
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            role_access._verify_token("invalid_token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid or expired token" in str(exc_info.value.message)
    
    def test_check_role_valid(self):
        """Test 7: Check role hợp lệ"""
        # Arrange
        payload = {"role": "ADMIN"}
        allowed_roles = [AccountsRole.ADMIN, AccountsRole.AGENT]
        
        # Act & Assert (không raise exception)
        self.role_access._check_role(payload, allowed_roles)
    
    def test_check_role_string_to_enum(self):
        """Test 8: Convert string role to enum"""
        # Arrange
        payload = {"role": "AGENT"}
        allowed_roles = [AccountsRole.AGENT]
        
        # Act & Assert (không raise exception)
        self.role_access._check_role(payload, allowed_roles)
    
    def test_check_role_invalid(self):
        """Test 9: Role không được phép (403)"""
        # Arrange
        payload = {"role": "CUSTOMER"}
        allowed_roles = [AccountsRole.ADMIN, AccountsRole.AGENT]
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            self.role_access._check_role(payload, allowed_roles)
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in str(exc_info.value.message)
    
    def test_check_role_missing_in_payload(self):
        """Test 10: Role không tồn tại trong payload"""
        # Arrange
        payload = {"username": "test"}
        allowed_roles = [AccountsRole.ADMIN]
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            self.role_access._check_role(payload, allowed_roles)
        
        assert exc_info.value.status_code == 401
        assert "Role not found in token" in str(exc_info.value.message)
    
    def test_check_role_invalid_enum_value(self):
        """Test 11: Invalid enum value"""
        # Arrange
        payload = {"role": "INVALID_ROLE"}
        allowed_roles = [AccountsRole.ADMIN]
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            self.role_access._check_role(payload, allowed_roles)
        
        assert exc_info.value.status_code == 401
        assert "Invalid role in token" in str(exc_info.value.message)
    
    def test_get_current_user_success(self):
        """Test 12: Get current user thành công"""
        # Arrange
        request = Mock(spec=Request)
        request.state.user = {
            "username": "testuser",
            "role": "ADMIN",
            "user_id": "123"
        }
        
        # Act
        user = get_current_user(request)
        
        # Assert
        assert user["username"] == "testuser"
        assert user["role"] == "ADMIN"
        assert user["user_id"] == "123"
    
    def test_get_current_user_not_authenticated(self):
        """Test 13: Get current user khi chưa authenticate"""
        # Arrange
        request = Mock(spec=Request)
        # Không set request.state.user
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            get_current_user(request)
        
        assert exc_info.value.status_code == 401
        assert "not authenticated" in str(exc_info.value.message).lower()
    
    def test_get_current_role_success(self):
        """Test 14: Get current role thành công"""
        # Arrange
        request = Mock(spec=Request)
        request.state.user = {
            "username": "testuser",
            "role": "AGENT",
            "user_id": "123"
        }
        
        # Act
        role = get_current_role(request)
        
        # Assert
        assert role == AccountsRole.AGENT
    
    def test_get_current_user_id_success(self):
        """Test 15: Get current user_id thành công"""
        # Arrange
        request = Mock(spec=Request)
        request.state.user = {
            "username": "testuser",
            "role": "CUSTOMER",
            "user_id": "uuid-123-456"
        }
        
        # Act
        user_id = get_current_user_id(request)
        
        # Assert
        assert user_id == "uuid-123-456"
    
    def test_get_current_user_id_missing(self):
        """Test 16: Get current user_id khi missing trong token"""
        # Arrange
        request = Mock(spec=Request)
        request.state.user = {
            "username": "testuser",
            "role": "CUSTOMER"
            # Không có user_id
        }
        
        # Act & Assert
        with pytest.raises(APIException) as exc_info:
            get_current_user_id(request)
        
        assert exc_info.value.status_code == 401
        assert "User ID not found" in str(exc_info.value.message)


class TestDecoratorIntegration:
    """Integration tests cho decorator"""
    
    @patch('src.Features.AuthAPI.RoleBasedAccess.JWTProvider')
    def test_decorator_valid_token_and_role(self, mock_jwt_provider_class):
        """Test 17: Decorator với token và role hợp lệ"""
        # Arrange
        mock_jwt_provider = Mock()
        mock_jwt_provider.verify_token.return_value = {
            "username": "admin",
            "role": "ADMIN",
            "user_id": "123"
        }
        mock_jwt_provider_class.return_value = mock_jwt_provider
        
        role_access = RoleBasedAccess(mock_jwt_provider)
        
        async def mock_endpoint(request):
            return {"success": True}
        
        request = Mock(spec=Request)
        request.headers = Headers({"Authorization": "Bearer valid_token"})
        request.state = Mock()
        
        decorated = role_access.require_role(AccountsRole.ADMIN)(mock_endpoint)
        
        # Act
        result = None
        import asyncio
        result = asyncio.run(decorated(request))
        
        # Assert
        assert result == {"success": True}
    
    @patch('src.Features.AuthAPI.RoleBasedAccess.JWTProvider')
    def test_decorator_wrong_role(self, mock_jwt_provider_class):
        """Test 18: Decorator với role sai (403)"""
        # Arrange
        mock_jwt_provider = Mock()
        mock_jwt_provider.verify_token.return_value = {
            "username": "customer",
            "role": "CUSTOMER",
            "user_id": "456"
        }
        mock_jwt_provider_class.return_value = mock_jwt_provider
        
        role_access = RoleBasedAccess(mock_jwt_provider)
        
        async def mock_endpoint(request):
            return {"success": True}
        
        request = Mock(spec=Request)
        request.headers = Headers({"Authorization": "Bearer valid_token"})
        request.state = Mock()
        
        decorated = role_access.require_role(AccountsRole.ADMIN)(mock_endpoint)
        
        # Act & Assert
        import asyncio
        with pytest.raises(APIException) as exc_info:
            asyncio.run(decorated(request))
        
        assert exc_info.value.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
