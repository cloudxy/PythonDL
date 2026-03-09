import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { authApi } from '@/api/auth'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || '')
  const refreshToken = ref(localStorage.getItem('refreshToken') || '')
  const user = ref(null)

  const isAuthenticated = computed(() => !!token.value)

  const login = async (credentials) => {
    try {
      const response = await authApi.login(credentials.username, credentials.password)
      
      if (response.access_token) {
        token.value = response.access_token
        refreshToken.value = response.refresh_token
        localStorage.setItem('token', response.access_token)
        localStorage.setItem('refreshToken', response.refresh_token)
      }
      
      return response
    } catch (error) {
      throw error
    }
  }

  const logout = async () => {
    try {
      await authApi.logout()
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      token.value = ''
      refreshToken.value = ''
      user.value = null
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
    }
  }

  const fetchUser = async () => {
    if (!token.value) return null
    
    try {
      const response = await authApi.getCurrentUser()
      user.value = response
      return response
    } catch (error) {
      logout()
      throw error
    }
  }

  const refreshAccessToken = async () => {
    if (!refreshToken.value) {
      logout()
      return null
    }
    
    try {
      const response = await authApi.refreshToken(refreshToken.value)
      
      if (response.access_token) {
        token.value = response.access_token
        refreshToken.value = response.refresh_token
        localStorage.setItem('token', response.access_token)
        localStorage.setItem('refreshToken', response.refresh_token)
      }
      
      return response
    } catch (error) {
      logout()
      throw error
    }
  }

  const initAuth = async () => {
    if (token.value) {
      try {
        await fetchUser()
      } catch (error) {
        console.error('Init auth error:', error)
      }
    }
  }

  return {
    token,
    refreshToken,
    user,
    isAuthenticated,
    login,
    logout,
    fetchUser,
    refreshAccessToken,
    initAuth
  }
})
