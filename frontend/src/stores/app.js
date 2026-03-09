import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  const sidebarCollapsed = ref(false)
  const theme = ref(localStorage.getItem('theme') || 'light')
  const loading = ref(false)
  const notifications = ref([])

  const toggleSidebar = () => {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  const setTheme = (newTheme) => {
    theme.value = newTheme
    localStorage.setItem('theme', newTheme)
    document.documentElement.classList.toggle('dark', newTheme === 'dark')
  }

  const showLoading = () => {
    loading.value = true
  }

  const hideLoading = () => {
    loading.value = false
  }

  const addNotification = (notification) => {
    const id = Date.now()
    notifications.value.push({
      id,
      ...notification
    })
    
    // 自动移除
    setTimeout(() => {
      removeNotification(id)
    }, notification.duration || 5000)
    
    return id
  }

  const removeNotification = (id) => {
    const index = notifications.value.findIndex(n => n.id === id)
    if (index > -1) {
      notifications.value.splice(index, 1)
    }
  }

  const success = (message) => {
    return addNotification({ type: 'success', message })
  }

  const error = (message) => {
    return addNotification({ type: 'error', message })
  }

  const warning = (message) => {
    return addNotification({ type: 'warning', message })
  }

  const info = (message) => {
    return addNotification({ type: 'info', message })
  }

  const initApp = () => {
    // 初始化主题
    if (theme.value === 'dark') {
      document.documentElement.classList.add('dark')
    }
  }

  return {
    sidebarCollapsed,
    theme,
    loading,
    notifications,
    toggleSidebar,
    setTheme,
    showLoading,
    hideLoading,
    addNotification,
    removeNotification,
    success,
    error,
    warning,
    info,
    initApp
  }
})
