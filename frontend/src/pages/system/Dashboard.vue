<template>
  <div class="space-y-6">
    <PageHeader title="仪表盘" subtitle="系统概览和关键指标" />

    <!-- 统计卡片 -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard
        label="总用户数"
        :value="stats.totalUsers"
        :change="stats.userGrowth"
        prefix=""
        color="primary"
      />
      <StatCard
        label="活跃用户"
        :value="stats.activeUsers"
        :change="stats.activeGrowth"
        color="success"
      />
      <StatCard
        label="今日访问"
        :value="stats.todayVisits"
        :change="stats.visitGrowth"
        color="warning"
      />
      <StatCard
        label="系统任务"
        :value="stats.totalTasks"
        :change="stats.taskGrowth"
        color="danger"
      />
    </div>

    <!-- 图表区域 -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- 访问趋势 -->
      <Card title="访问趋势">
        <div class="h-64 flex items-center justify-center text-secondary-400">
          <div class="text-center">
            <svg class="w-16 h-16 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
            <p>访问趋势图表</p>
            <p class="text-sm mt-1">集成 Chart.js 后显示</p>
          </div>
        </div>
      </Card>

      <!-- 模块使用统计 -->
      <Card title="模块使用统计">
        <div class="space-y-4">
          <ProgressBar label="金融分析" :value="75" :max="100" color="primary" />
          <ProgressBar label="气象分析" :value="60" :max="100" color="success" />
          <ProgressBar label="消费分析" :value="45" :max="100" color="warning" />
          <ProgressBar label="看相算命" :value="30" :max="100" color="danger" />
          <ProgressBar label="爬虫采集" :value="20" :max="100" color="secondary" />
        </div>
      </Card>
    </div>

    <!-- 最近活动和系统状态 -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 最近活动 -->
      <ActivityList
        title="最近活动"
        :items="recentActivities"
        show-more
      />

      <!-- 系统状态 -->
      <Card title="系统状态">
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-sm text-secondary-600">CPU 使用率</span>
            <span class="text-sm font-medium text-secondary-900">45%</span>
          </div>
          <Progress :value="45" color="primary" />
          
          <div class="flex items-center justify-between">
            <span class="text-sm text-secondary-600">内存使用率</span>
            <span class="text-sm font-medium text-secondary-900">68%</span>
          </div>
          <Progress :value="68" color="warning" />
          
          <div class="flex items-center justify-between">
            <span class="text-sm text-secondary-600">磁盘使用率</span>
            <span class="text-sm font-medium text-secondary-900">35%</span>
          </div>
          <Progress :value="35" color="success" />
        </div>
      </Card>

      <!-- 快捷操作 -->
      <Card title="快捷操作">
        <div class="grid grid-cols-2 gap-3">
          <router-link to="/system/users" class="p-4 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors text-center">
            <svg class="w-8 h-8 mx-auto text-primary-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
            <span class="text-sm font-medium text-primary-700">用户管理</span>
          </router-link>
          <router-link to="/system/roles" class="p-4 bg-success-50 rounded-lg hover:bg-success-100 transition-colors text-center">
            <svg class="w-8 h-8 mx-auto text-success-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span class="text-sm font-medium text-success-700">角色管理</span>
          </router-link>
          <router-link to="/system/config" class="p-4 bg-warning-50 rounded-lg hover:bg-warning-100 transition-colors text-center">
            <svg class="w-8 h-8 mx-auto text-warning-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span class="text-sm font-medium text-warning-700">系统配置</span>
          </router-link>
          <router-link to="/crawler" class="p-4 bg-danger-50 rounded-lg hover:bg-danger-100 transition-colors text-center">
            <svg class="w-8 h-8 mx-auto text-danger-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span class="text-sm font-medium text-danger-700">爬虫任务</span>
          </router-link>
        </div>
      </Card>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import StatCard from '@/components/ui/StatCard.vue'
import Card from '@/components/ui/Card.vue'
import Progress from '@/components/ui/Progress.vue'
import ProgressBar from '@/components/ui/ProgressBar.vue'
import ActivityList from '@/components/ui/ActivityList.vue'

const stats = ref({
  totalUsers: 1256,
  userGrowth: 12.5,
  activeUsers: 892,
  activeGrowth: 8.3,
  todayVisits: 3456,
  visitGrowth: 15.2,
  totalTasks: 48,
  taskGrowth: -2.1
})

const recentActivities = ref([
  {
    title: '用户登录',
    description: '管理员登录系统',
    time: '5分钟前',
    icon: 'login',
    iconBg: 'bg-primary-100',
    iconClass: 'text-primary-600'
  },
  {
    title: '数据同步',
    description: '股票数据同步完成',
    time: '15分钟前',
    icon: 'sync',
    iconBg: 'bg-success-100',
    iconClass: 'text-success-600'
  },
  {
    title: '系统警告',
    description: '内存使用率超过80%',
    time: '1小时前',
    icon: 'warning',
    iconBg: 'bg-warning-100',
    iconClass: 'text-warning-600'
  },
  {
    title: '任务完成',
    description: '气象数据采集任务完成',
    time: '2小时前',
    icon: 'check',
    iconBg: 'bg-success-100',
    iconClass: 'text-success-600'
  }
])

onMounted(() => {
  // 加载数据
})
</script>
