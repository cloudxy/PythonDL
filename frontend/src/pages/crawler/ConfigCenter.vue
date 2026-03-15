<template>
  <div class="space-y-6">
    <PageHeader title="爬虫配置中心" subtitle="管理爬虫配置和数据流">
      <template #actions>
        <Button variant="primary" @click="showCreateModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          新建爬虫配置
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '爬虫采集', to: '/crawler' },
      { label: '配置中心' }
    ]" />

    <!-- 爬虫类型选择 -->
    <Card>
      <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div
          v-for="type in crawlerTypes"
          :key="type.value"
          :class="selectedType === type.value ? 'border-primary-500 bg-primary-50' : 'border-secondary-200 hover:border-primary-300'"
          class="border rounded-lg p-4 cursor-pointer transition-all"
          @click="selectedType = type.value"
        >
          <div class="text-3xl mb-2">{{ type.icon }}</div>
          <div class="font-semibold">{{ type.name }}</div>
          <div class="text-xs text-secondary-500 mt-1">{{ type.description }}</div>
        </div>
      </div>
    </Card>

    <!-- 配置详情 -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- 数据源配置 -->
      <Card title="数据源配置">
        <div class="space-y-4">
          <div v-for="(source, index) in currentProfile.data_sources" :key="index"
               class="border rounded-lg p-4 hover:shadow-md transition-all">
            <div class="flex items-center justify-between mb-2">
              <h4 class="font-semibold text-primary-600">{{ source.name }}</h4>
              <Badge :variant="source.enabled ? 'success' : 'secondary'">
                {{ source.enabled ? '启用' : '禁用' }}
              </Badge>
            </div>
            
            <div class="text-sm space-y-2">
              <div class="flex items-center gap-2">
                <span class="text-secondary-500 w-16">URL:</span>
                <code class="text-xs bg-secondary-100 px-2 py-1 rounded flex-1">{{ source.url }}</code>
              </div>
              
              <div class="flex items-center gap-2">
                <span class="text-secondary-500 w-16">方法:</span>
                <Badge variant="info">{{ source.method }}</Badge>
              </div>
              
              <div class="flex items-center gap-2">
                <span class="text-secondary-500 w-16">缓存:</span>
                <span>{{ formatDuration(source.cache_ttl) }}</span>
              </div>
              
              <div v-if="source.auth" class="flex items-center gap-2">
                <span class="text-secondary-500 w-16">认证:</span>
                <Badge :variant="source.auth.auth_type === 'api_key' ? 'warning' : 'info'">
                  {{ getAuthTypeName(source.auth.auth_type) }}
                </Badge>
              </div>
              
              <div class="border-t mt-2 pt-2">
                <div class="text-secondary-500 text-xs mb-1">采集字段 ({{ source.fields?.length || 0 }}个):</div>
                <div class="flex flex-wrap gap-1">
                  <Badge v-for="field in source.fields?.slice(0, 5)" :key="field.field_name" variant="secondary" class="text-xs">
                    {{ field.field_name }}
                  </Badge>
                  <Badge v-if="source.fields?.length > 5" variant="secondary" class="text-xs">
                    +{{ source.fields.length - 5 }}
                  </Badge>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      <!-- 采集频率配置 -->
      <Card title="采集频率与调度">
        <div class="space-y-4">
          <!-- 频率限制 -->
          <div class="border rounded-lg p-4">
            <h4 class="font-semibold mb-3 flex items-center gap-2">
              <svg class="w-5 h-5 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              频率限制
            </h4>
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div class="bg-secondary-50 p-2 rounded">
                <div class="text-secondary-500 text-xs">每秒请求</div>
                <div class="font-semibold text-lg">{{ currentProfile.rate_limit?.requests_per_second || 1 }}</div>
              </div>
              <div class="bg-secondary-50 p-2 rounded">
                <div class="text-secondary-500 text-xs">每分钟请求</div>
                <div class="font-semibold text-lg">{{ currentProfile.rate_limit?.requests_per_minute || 60 }}</div>
              </div>
              <div class="bg-secondary-50 p-2 rounded">
                <div class="text-secondary-500 text-xs">并发限制</div>
                <div class="font-semibold text-lg">{{ currentProfile.rate_limit?.concurrent_limit || 5 }}</div>
              </div>
              <div class="bg-secondary-50 p-2 rounded">
                <div class="text-secondary-500 text-xs">突发大小</div>
                <div class="font-semibold text-lg">{{ currentProfile.rate_limit?.burst_size || 10 }}</div>
              </div>
            </div>
          </div>

          <!-- 调度配置 -->
          <div class="border rounded-lg p-4">
            <h4 class="font-semibold mb-3 flex items-center gap-2">
              <svg class="w-5 h-5 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              定时调度
            </h4>
            <div class="space-y-2 text-sm">
              <div class="flex items-center justify-between">
                <span class="text-secondary-500">状态:</span>
                <Badge :variant="currentProfile.schedule?.enabled ? 'success' : 'secondary'">
                  {{ currentProfile.schedule?.enabled ? '已启用' : '已禁用' }}
                </Badge>
              </div>
              <div class="flex items-center justify-between">
                <span class="text-secondary-500">Cron 表达式:</span>
                <code class="bg-secondary-100 px-2 py-1 rounded">{{ currentProfile.schedule?.cron_expression || '未设置' }}</code>
              </div>
              <div class="flex items-center justify-between">
                <span class="text-secondary-500">执行间隔:</span>
                <span>{{ formatDuration(currentProfile.schedule?.interval_seconds || 0) }}</span>
              </div>
              <div class="flex items-center justify-between">
                <span class="text-secondary-500">执行时间:</span>
                <div class="flex gap-1">
                  <Badge v-for="time in currentProfile.schedule?.execute_at" :key="time" variant="info" class="text-xs">
                    {{ time }}
                  </Badge>
                </div>
              </div>
            </div>
          </div>

          <!-- 数据管道 -->
          <div class="border rounded-lg p-4">
            <h4 class="font-semibold mb-3 flex items-center gap-2">
              <svg class="w-5 h-5 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
              数据管道
            </h4>
            <div class="space-y-2">
              <div v-for="(step, index) in currentProfile.data_pipeline" :key="index"
                   class="flex items-center gap-2 text-sm">
                <div class="w-6 h-6 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center text-xs font-semibold">
                  {{ index + 1 }}
                </div>
                <Badge :variant="getPipelineStepVariant(step.type)">{{ getPipelineStepName(step.type) }}</Badge>
                <span class="text-secondary-600">{{ step.name }}</span>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>

    <!-- 采集字段配置 -->
    <Card title="采集字段配置">
      <div class="overflow-x-auto">
        <Table :columns="fieldColumns" :data="allFields">
          <template #cell-required="{ row }">
            <Badge v-if="row.required" variant="danger" class="text-xs">必填</Badge>
            <span v-else class="text-secondary-400">可选</span>
          </template>
          <template #cell-field_type="{ row }">
            <Badge :variant="getFieldTypeVariant(row.field_type)" class="text-xs">
              {{ row.field_type }}
            </Badge>
          </template>
        </Table>
      </div>
    </Card>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Badge from '@/components/ui/Badge.vue'
import Table from '@/components/ui/Table.vue'
import { api } from '@/api'

const selectedType = ref('stock')
const loading = ref(false)
const showCreateModal = ref(false)

const crawlerTypes = [
  { value: 'stock', name: '股票爬虫', icon: '📈', description: '股票行情数据采集' },
  { value: 'weather', name: '气象爬虫', icon: '🌤️', description: '气象数据采集' },
  { value: 'consumption', name: '消费爬虫', icon: '💰', description: '宏观经济数据' },
  { value: 'fortune', name: '算命爬虫', icon: '🔮', description: '周易算命数据' },
  { value: 'general', name: '通用爬虫', icon: '🔧', description: '通用网页采集' }
]

const fieldColumns = [
  { key: 'field_name', label: '字段名' },
  { key: 'field_type', label: '类型' },
  { key: 'source_path', label: '来源路径' },
  { key: 'required', label: '必填' },
  { key: 'description', label: '描述' }
]

const currentProfile = reactive({
  data_sources: [],
  rate_limit: {},
  schedule: {},
  data_pipeline: []
})

const allFields = computed(() => {
  const fields = []
  currentProfile.data_sources.forEach(source => {
    if (source.fields) {
      source.fields.forEach(field => {
        fields.push({
          ...field,
          source_name: source.name
        })
      })
    }
  })
  return fields
})

const getAuthTypeName = (type) => {
  const names = {
    'none': '无认证',
    'api_key': 'API Key',
    'token': 'Token',
    'oauth2': 'OAuth2',
    'basic': 'Basic Auth',
    'cookie': 'Cookie'
  }
  return names[type] || type
}

const getPipelineStepName = (type) => {
  const names = {
    'filter': '过滤',
    'transform': '转换',
    'validate': '验证',
    'enrich': '增强',
    'save': '存储'
  }
  return names[type] || type
}

const getPipelineStepVariant = (type) => {
  const variants = {
    'filter': 'secondary',
    'transform': 'info',
    'validate': 'warning',
    'enrich': 'primary',
    'save': 'success'
  }
  return variants[type] || 'secondary'
}

const getFieldTypeVariant = (type) => {
  const variants = {
    'string': 'secondary',
    'number': 'info',
    'date': 'warning',
    'boolean': 'success',
    'json': 'primary'
  }
  return variants[type] || 'secondary'
}

const formatDuration = (seconds) => {
  if (!seconds) return '-'
  if (seconds < 60) return `${seconds}秒`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}分钟`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}小时`
  return `${Math.floor(seconds / 86400)}天`
}

const loadProfile = async (type) => {
  loading.value = true
  try {
    const response = await api.get(`/crawler-display/config/${type}`)
    if (response.success) {
      Object.assign(currentProfile, {
        data_sources: response.data.default_data_sources || [],
        rate_limit: response.data.rate_limit || {},
        schedule: response.data.schedule || {},
        data_pipeline: response.data.data_pipeline || []
      })
    }
  } catch (error) {
    console.error('加载配置失败:', error)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadProfile(selectedType.value)
})

// 监听类型变化
watch(selectedType, (newType) => {
  loadProfile(newType)
})
</script>
