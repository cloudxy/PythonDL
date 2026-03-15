import { createRouter, createWebHistory } from 'vue-router'
import AuthLayout from '@/layouts/AuthLayout.vue'
import MainLayout from '@/layouts/MainLayout.vue'

const routes = [
  // 认证页面
  {
    path: '/',
    component: AuthLayout,
    children: [
      {
        path: '',
        redirect: '/login'
      },
      {
        path: 'login',
        name: 'Login',
        component: () => import('@/pages/auth/Login.vue'),
        meta: { title: '登录' }
      },
      {
        path: 'forgot-password',
        name: 'ForgotPassword',
        component: () => import('@/pages/auth/ForgotPassword.vue'),
        meta: { title: '找回密码' }
      }
    ]
  },

  // 主应用页面
  {
    path: '/',
    component: MainLayout,
    meta: { requiresAuth: true },
    children: [
      // 仪表盘
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/pages/system/Dashboard.vue'),
        meta: { title: '仪表盘' }
      },

      // 系统管理
      {
        path: 'system/users',
        name: 'UserList',
        component: () => import('@/pages/system/UserList.vue'),
        meta: { title: '用户管理' }
      },
      {
        path: 'system/roles',
        name: 'RoleList',
        component: () => import('@/pages/system/RoleList.vue'),
        meta: { title: '角色管理' }
      },
      {
        path: 'system/permissions',
        name: 'PermissionList',
        component: () => import('@/pages/system/PermissionList.vue'),
        meta: { title: '权限管理' }
      },
      {
        path: 'system/config',
        name: 'ConfigList',
        component: () => import('@/pages/system/ConfigList.vue'),
        meta: { title: '系统配置' }
      },
      {
        path: 'system/logs',
        name: 'LogList',
        component: () => import('@/pages/system/LogList.vue'),
        meta: { title: '操作日志' }
      },

      // 金融分析
      {
        path: 'finance/stocks',
        name: 'StockList',
        component: () => import('@/pages/finance/StockList.vue'),
        meta: { title: '股票管理' }
      },
      {
        path: 'finance/prediction',
        name: 'StockPrediction',
        component: () => import('@/pages/finance/StockPrediction.vue'),
        meta: { title: '股票预测' }
      },
      {
        path: 'finance/risk',
        name: 'RiskAssessment',
        component: () => import('@/pages/finance/RiskAssessment.vue'),
        meta: { title: '风险评估' }
      },

      // 气象分析
      {
        path: 'weather/stations',
        name: 'WeatherStationList',
        component: () => import('@/pages/weather/StationList.vue'),
        meta: { title: '气象站点' }
      },
      {
        path: 'weather/data',
        name: 'WeatherDataList',
        component: () => import('@/pages/weather/DataList.vue'),
        meta: { title: '气象数据' }
      },
      {
        path: 'weather/prediction',
        name: 'WeatherPrediction',
        component: () => import('@/pages/weather/Prediction.vue'),
        meta: { title: '气象预测' }
      },

      // 看相算命
      {
        path: 'fortune/fengshui',
        name: 'FengShuiList',
        component: () => import('@/pages/fortune/FengShuiList.vue'),
        meta: { title: '风水数据' }
      },
      {
        path: 'fortune/face',
        name: 'FaceReadingList',
        component: () => import('@/pages/fortune/FaceReadingList.vue'),
        meta: { title: '面相数据' }
      },
      {
        path: 'fortune/bazi',
        name: 'BaziList',
        component: () => import('@/pages/fortune/BaziList.vue'),
        meta: { title: '八字数据' }
      },
      {
        path: 'fortune/zhouyi',
        name: 'ZhouYiList',
        component: () => import('@/pages/fortune/ZhouYiList.vue'),
        meta: { title: '周易数据' }
      },
      {
        path: 'fortune/constellation',
        name: 'ConstellationList',
        component: () => import('@/pages/fortune/ConstellationList.vue'),
        meta: { title: '星座数据' }
      },
      {
        path: 'fortune/luck',
        name: 'LuckList',
        component: () => import('@/pages/fortune/LuckList.vue'),
        meta: { title: '运势数据' }
      },
      {
        path: 'fortune/analysis',
        name: 'FortuneAnalysis',
        component: () => import('@/pages/fortune/AnalysisPage.vue'),
        meta: { title: '综合分析' }
      },

      // 消费分析
      {
        path: 'consumption/gdp',
        name: 'GdpList',
        component: () => import('@/pages/consumption/GdpList.vue'),
        meta: { title: 'GDP数据' }
      },
      {
        path: 'consumption/population',
        name: 'PopulationList',
        component: () => import('@/pages/consumption/PopulationList.vue'),
        meta: { title: '人口数据' }
      },
      {
        path: 'consumption/indicators',
        name: 'IndicatorList',
        component: () => import('@/pages/consumption/IndicatorList.vue'),
        meta: { title: '经济指标' }
      },
      {
        path: 'consumption/community',
        name: 'CommunityList',
        component: () => import('@/pages/consumption/CommunityList.vue'),
        meta: { title: '小区数据' }
      },
      {
        path: 'consumption/prediction',
        name: 'ConsumptionPrediction',
        component: () => import('@/pages/consumption/Prediction.vue'),
        meta: { title: '消费预测' }
      },

      // 爬虫采集
      {
        path: 'crawler',
        name: 'CrawlerTaskList',
        component: () => import('@/pages/crawler/TaskList.vue'),
        meta: { title: '爬虫采集' }
      },
      {
        path: 'crawler/ai',
        name: 'AICrawlerTaskList',
        component: () => import('@/pages/crawler/AITaskList.vue'),
        meta: { title: 'AI 智能爬虫' }
      },
      {
        path: 'crawler/display',
        name: 'CrawlerDataDisplay',
        component: () => import('@/pages/crawler/DataDisplay.vue'),
        meta: { title: '数据展示' }
      },
      {
        path: 'crawler/config',
        name: 'CrawlerConfigCenter',
        component: () => import('@/pages/crawler/ConfigCenter.vue'),
        meta: { title: '配置中心' }
      },

      // 404页面
      {
        path: '404',
        name: 'NotFound',
        component: () => import('@/pages/NotFound.vue'),
        meta: { title: '页面未找到' }
      }
    ]
  },

  // 未知路由重定向
  {
    path: '/:pathMatch(.*)*',
    redirect: '/404'
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫
router.beforeEach((to, from, next) => {
  // 设置页面标题
  document.title = to.meta.title ? `${to.meta.title} - PythonDL` : 'PythonDL 智能分析平台'

  // 检查是否需要认证
  const token = localStorage.getItem('token')
  if (to.meta.requiresAuth && !token) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
  } else if ((to.name === 'Login' || to.name === 'ForgotPassword') && token) {
    next({ name: 'Dashboard' })
  } else {
    next()
  }
})

export default router
