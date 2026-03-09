<template>
  <div class="min-h-screen bg-secondary-50 flex">
    <!-- 侧边栏 -->
    <aside 
      :class="[
        'fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-secondary-200 transform transition-transform duration-300 lg:translate-x-0 lg:static lg:inset-0',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      ]"
    >
      <!-- Logo -->
      <div class="h-16 flex items-center px-6 border-b border-secondary-200">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
            <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <span class="text-lg font-bold text-secondary-900">PythonDL</span>
        </div>
      </div>

      <!-- 导航菜单 -->
      <nav class="flex-1 overflow-y-auto py-4">
        <template v-for="item in menuItems" :key="item.path">
          <!-- 有子菜单 -->
          <div v-if="item.children" class="mb-1">
            <button
              @click="toggleSubmenu(item.path)"
              :class="[
                'w-full flex items-center justify-between px-4 py-3 text-secondary-600 hover:bg-primary-50 hover:text-primary-600 transition-colors',
                isSubmenuOpen(item.path) ? 'bg-primary-50 text-primary-600' : ''
              ]"
            >
              <div class="flex items-center gap-3">
                <component :is="item.icon" class="w-5 h-5" />
                <span class="font-medium">{{ item.name }}</span>
              </div>
              <svg
                :class="['w-4 h-4 transition-transform', isSubmenuOpen(item.path) ? 'rotate-180' : '']"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div v-show="isSubmenuOpen(item.path)" class="bg-secondary-50 py-1">
              <router-link
                v-for="child in item.children"
                :key="child.path"
                :to="child.path"
                class="flex items-center gap-3 px-4 py-2.5 text-sm text-secondary-600 hover:bg-primary-50 hover:text-primary-600 transition-colors"
                :class="{ 'text-primary-600 bg-primary-50': isActiveRoute(child.path) }"
              >
                <span class="w-1.5 h-1.5 rounded-full bg-current"></span>
                {{ child.name }}
              </router-link>
            </div>
          </div>
          
          <!-- 无子菜单 -->
          <router-link
            v-else
            :to="item.path"
            class="flex items-center gap-3 px-4 py-3 text-secondary-600 hover:bg-primary-50 hover:text-primary-600 transition-colors"
            :class="{ 'bg-primary-50 text-primary-600 border-r-2 border-primary-600': isActiveRoute(item.path) }"
          >
            <component :is="item.icon" class="w-5 h-5" />
            <span class="font-medium">{{ item.name }}</span>
          </router-link>
        </template>
      </nav>

      <!-- 用户信息 -->
      <div class="border-t border-secondary-200 p-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
            <span class="text-primary-600 font-semibold">{{ userInitials }}</span>
          </div>
          <div class="flex-1 min-w-0">
            <p class="text-sm font-medium text-secondary-900 truncate">{{ user?.username || '用户' }}</p>
            <p class="text-xs text-secondary-500 truncate">{{ user?.email || '' }}</p>
          </div>
        </div>
      </div>
    </aside>

    <!-- 移动端遮罩 -->
    <div
      v-show="sidebarOpen"
      @click="sidebarOpen = false"
      class="fixed inset-0 bg-black/50 z-40 lg:hidden"
    ></div>

    <!-- 主内容区域 -->
    <div class="flex-1 flex flex-col min-w-0">
      <!-- 顶部导航 -->
      <header class="h-16 bg-white border-b border-secondary-200 flex items-center justify-between px-4 lg:px-6 sticky top-0 z-30">
        <!-- 左侧 -->
        <div class="flex items-center gap-4">
          <button
            @click="sidebarOpen = !sidebarOpen"
            class="lg:hidden p-2 text-secondary-600 hover:bg-secondary-100 rounded-lg"
          >
            <svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h1 class="text-xl font-semibold text-secondary-900">{{ pageTitle }}</h1>
        </div>

        <!-- 右侧 -->
        <div class="flex items-center gap-3">
          <!-- 搜索 -->
          <button class="p-2 text-secondary-500 hover:bg-secondary-100 rounded-lg hidden sm:block">
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>

          <!-- 通知 -->
          <button class="p-2 text-secondary-500 hover:bg-secondary-100 rounded-lg relative">
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            <span class="absolute top-1.5 right-1.5 w-2 h-2 bg-danger-500 rounded-full"></span>
          </button>

          <!-- 用户菜单 -->
          <div class="relative" ref="userMenuRef">
            <button
              @click="showUserMenu = !showUserMenu"
              class="flex items-center gap-2 p-1.5 hover:bg-secondary-100 rounded-lg"
            >
              <div class="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                <span class="text-primary-600 text-sm font-semibold">{{ userInitials }}</span>
              </div>
              <svg class="w-4 h-4 text-secondary-500 hidden sm:block" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            <!-- 下拉菜单 -->
            <Transition
              enter-active-class="transition ease-out duration-100"
              enter-from-class="transform opacity-0 scale-95"
              enter-to-class="transform opacity-100 scale-100"
              leave-active-class="transition ease-in duration-75"
              leave-from-class="transform opacity-100 scale-100"
              leave-to-class="transform opacity-0 scale-95"
            >
              <div v-show="showUserMenu" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-secondary-100 py-1 z-50">
                <router-link to="/profile" class="flex items-center gap-2 px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-50">
                  <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  个人中心
                </router-link>
                <router-link to="/settings" class="flex items-center gap-2 px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-50">
                  <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  系统设置
                </router-link>
                <hr class="my-1 border-secondary-100" />
                <button @click="handleLogout" class="w-full flex items-center gap-2 px-4 py-2 text-sm text-danger-600 hover:bg-danger-50">
                  <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                  退出登录
                </button>
              </div>
            </Transition>
          </div>
        </div>
      </header>

      <!-- 页面内容 -->
      <main class="flex-1 p-4 lg:p-6 overflow-auto">
        <router-view />
      </main>

      <!-- 页脚 -->
      <footer class="h-12 bg-white border-t border-secondary-200 flex items-center justify-center px-4">
        <p class="text-sm text-secondary-500">
          &copy; {{ currentYear }} PythonDL 智能分析平台. All rights reserved.
        </p>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, h } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const sidebarOpen = ref(false)
const showUserMenu = ref(false)
const userMenuRef = ref(null)
const openSubmenus = ref(['/system', '/finance', '/weather', '/fortune', '/consumption'])

const user = computed(() => authStore.user)
const userInitials = computed(() => {
  if (user.value?.username) {
    return user.value.username.slice(0, 2).toUpperCase()
  }
  return 'U'
})

const currentYear = computed(() => new Date().getFullYear())

const pageTitle = computed(() => {
  const title = route.meta?.title
  return title || 'PythonDL'
})

// 图标组件
const DashboardIcon = {
  render() {
    return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z' })
    ])
  }
}

const SystemIcon = {
  render() {
    return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z' }),
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M15 12a3 3 0 11-6 0 3 3 0 016 0z' })
    ])
  }
}

const FinanceIcon = {
  render() {
    return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' })
    ])
  }
}

const WeatherIcon = {
  render() {
    return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z' })
    ])
  }
}

const FortuneIcon = {
  render() {
    return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z' })
    ])
  }
}

const ConsumptionIcon = {
  render() {
    return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' })
    ])
  }
}

const CrawlerIcon = {
  render() {
    return h('svg', { class: 'w-5 h-5', fill: 'none', viewBox: '0 0 24 24', stroke: 'currentColor' }, [
      h('path', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': '2', d: 'M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15' })
    ])
  }
}

const menuItems = [
  { name: '仪表盘', path: '/dashboard', icon: DashboardIcon },
  {
    name: '系统管理',
    path: '/system',
    icon: SystemIcon,
    children: [
      { name: '用户管理', path: '/system/users' },
      { name: '角色管理', path: '/system/roles' },
      { name: '权限管理', path: '/system/permissions' },
      { name: '系统配置', path: '/system/config' },
      { name: '操作日志', path: '/system/logs' },
    ]
  },
  {
    name: '金融分析',
    path: '/finance',
    icon: FinanceIcon,
    children: [
      { name: '股票管理', path: '/finance/stocks' },
      { name: '股票预测', path: '/finance/prediction' },
      { name: '风险评估', path: '/finance/risk' },
    ]
  },
  {
    name: '气象分析',
    path: '/weather',
    icon: WeatherIcon,
    children: [
      { name: '气象站点', path: '/weather/stations' },
      { name: '气象数据', path: '/weather/data' },
      { name: '气象预测', path: '/weather/prediction' },
    ]
  },
  {
    name: '看相算命',
    path: '/fortune',
    icon: FortuneIcon,
    children: [
      { name: '风水数据', path: '/fortune/fengshui' },
      { name: '面相数据', path: '/fortune/face' },
      { name: '八字数据', path: '/fortune/bazi' },
      { name: '周易数据', path: '/fortune/zhouyi' },
      { name: '星座数据', path: '/fortune/constellation' },
      { name: '运势数据', path: '/fortune/luck' },
      { name: '综合分析', path: '/fortune/analysis' },
    ]
  },
  {
    name: '消费分析',
    path: '/consumption',
    icon: ConsumptionIcon,
    children: [
      { name: 'GDP数据', path: '/consumption/gdp' },
      { name: '人口数据', path: '/consumption/population' },
      { name: '经济指标', path: '/consumption/indicators' },
      { name: '小区数据', path: '/consumption/community' },
      { name: '消费预测', path: '/consumption/prediction' },
    ]
  },
  { name: '爬虫采集', path: '/crawler', icon: CrawlerIcon },
]

const isActiveRoute = (path) => {
  return route.path === path || route.path.startsWith(path + '/')
}

const isSubmenuOpen = (path) => {
  return openSubmenus.value.includes(path) || isActiveRoute(path)
}

const toggleSubmenu = (path) => {
  const index = openSubmenus.value.indexOf(path)
  if (index > -1) {
    openSubmenus.value.splice(index, 1)
  } else {
    openSubmenus.value.push(path)
  }
}

const handleLogout = async () => {
  await authStore.logout()
  router.push('/login')
}

const handleClickOutside = (event) => {
  if (userMenuRef.value && !userMenuRef.value.contains(event.target)) {
    showUserMenu.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>
