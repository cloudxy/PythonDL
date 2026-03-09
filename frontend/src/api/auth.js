import api from './index'

export const authApi = {
  login(username, password) {
    const formData = new URLSearchParams()
    formData.append('username', username)
    formData.append('password', password)
    
    return api.post('/v1/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    })
  },

  logout() {
    return api.post('/v1/auth/logout')
  },

  getCurrentUser() {
    return api.get('/v1/auth/me')
  },

  refreshToken(refreshToken) {
    const formData = new URLSearchParams()
    formData.append('refresh_token', refreshToken)
    
    return api.post('/v1/auth/refresh', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    })
  },

  forgotPassword(email) {
    return api.post('/v1/auth/forgot-password', { email })
  },

  resetPassword(data) {
    return api.post('/v1/auth/reset-password', data)
  },

  changePassword(data) {
    return api.post('/v1/auth/change-password', data)
  },

  register(data) {
    return api.post('/v1/auth/register', data)
  }
}
