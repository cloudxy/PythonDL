<template>
  <div class="space-y-6">
    <PageHeader title="仪表盘" subtitle="欢迎使用 PythonDL 智能分析平台">
      <template #actions>
        <Button variant="outline" @click="loadDashboard">
          <svg class="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          刷新
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '仪表盘' }
    ]" />

    <!-- 统计卡片 -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard title="用户总数" value="1,234" icon="👥" trend="+12%" />
      <StatCard title="股票数据" value="5,678" icon="📈" trend="+8%" />
      <StatCard title="气象站点" value="342" icon="🌤️" trend="+5%" />
      <StatCard title="采集任务" value="28" icon="🤖" trend="+15%" />
    </div>

    <!-- 图表区域 -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <template #header>
          <h3 class="text-lg font-semibold">数据趋势</h3>
        </template>
        <div class="h-64 bg-secondary-50 rounded-lg flex items-center justify-center text-secondary-400">
          <p>图表区域（集成图表库后显示）</p>
        </div>
      </Card>

      <Card>
        <template #header>
          <h3 class="text-lg font-semibold">系统负载</h3>
        </template>
        <div class="space-y-4">
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm text-secondary-600">CPU 使用率</span>
              <span class="text-sm font-medium">45%</span>
            </div>
            <div class="h-2 bg-secondary-200 rounded-full overflow-hidden">
              <div class="h-full bg-primary-500 rounded-full" style="width: 45%"></div>
            </div>
          </div>
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm text-secondary-600">内存使用率</span>
              <span class="text-sm font-medium">62%</span>
            </div>
            <div class="h-2 bg-secondary-200 rounded-full overflow-hidden">
              <div class="h-full bg-warning-500 rounded-full" style="width: 62%"></div>
            </div>
          </div>
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm text-secondary-600">磁盘使用率</span>
              <span class="text-sm font-medium">38%</span>
            </div>
            <div class="h-2 bg-secondary-200 rounded-full overflow-hidden">
              <div class="h-full bg-success-500 rounded-full" style="width: 38%"></div>
            </div>
          </div>
        </div>
      </Card>
    </div>

    <!-- 最近活动 -->
    <Card>
      <template #header>
        <h3 class="text-lg font-semibold">最近活动</h3>
      </template>
      <Timeline :items="activities" />
    </Card>

    <!-- 快捷入口 -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <router-link to="/finance/stocks" class="p-4 bg-white border border-secondary-200 rounded-lg hover:shadow-md transition-shadow text-center">
        <div class="text-2xl mb-2">📈</div>
        <div class="font-medium text-secondary-900">股票管理</div>
      </router-link>
      <router-link to="/weather/data" class="p-4 bg-white border border-secondary-200 rounded-lg hover:shadow-md transition-shadow text-center">
        <div class="text-2xl mb-2">🌤️</div>
        <div class="font-medium text-secondary-900">气象数据</div>
      </router-link>
      <router-link to="/fortune/analysis" class="p-4 bg-white border border-secondary-200 rounded-lg hover:shadow-md transition-shadow text-center">
        <div class="text-2xl mb-2">🔮</div>
        <div class="font-medium text-secondary-900">算命分析</div>
      </router-link>
      <router-link to="/crawler" class="p-4 bg-white border border-secondary-200 rounded-lg hover:shadow-md transition-shadow text-center">
        <div class="text-2xl mb-2">🤖</div>
        <div class="font-medium text-secondary-900">爬虫采集</div>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import StatCard from '@/components/ui/StatCard.vue'
import Timeline from '@/components/ui/Timeline.vue'

const loading = ref(false)

const activities = ref([
  { id: 1, time: '10:30', content: 'admin 登录了系统', icon: '🔐', type: 'success' },
  { id: 2, time: '10:25', content: '股票数据采集完成，新增 150 条数据', icon: '📈', type: 'info' },
  { id: 3, time: '10:20', content: '气象预测任务启动', icon: '🌤️', type: 'warning' },
  { id: 4, time: '10:15', content: '用户 user1 创建了新的爬虫任务', icon: '🤖', type: 'info' },
  { id: 5, time: '10:00', content: '系统定时备份完成', icon: '💾', type: 'success' }
])

const loadDashboard = async () => {
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 1000)
}

onMounted(() => {
  loadDashboard()
})
</script>
