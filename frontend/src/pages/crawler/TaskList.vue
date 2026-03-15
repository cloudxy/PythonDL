<template>
  <div class="space-y-6">
    <PageHeader title="爬虫采集" subtitle="管理数据采集任务">
      <template #actions>
        <Button variant="primary" @click="showCreateModal = true">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          新建任务
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '爬虫采集' }
    ]" />

    <Card>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard title="总任务数" :value="tasks.length.toString()" icon="📋" />
        <StatCard title="运行中" :value="runningCount.toString()" icon="⚙️" trend="实时" />
        <StatCard title="已完成" :value="completedCount.toString()" icon="✅" />
        <StatCard title="失败" :value="failedCount.toString()" icon="❌" trend="-2%" trend-negative />
      </div>

      <SearchBar
        v-model="searchQuery"
        placeholder="搜索任务名称、类型..."
        @search="handleSearch"
        @add="showCreateModal = true"
      />

      <Table
        :columns="columns"
        :data="filteredTasks"
        :loading="loading"
        :total="filteredTasks.length"
      >
        <template #cell-type="{ row }">
          <Badge :variant="getTypeBadgeVariant(row.type)">
            {{ typeMap[row.type] }}
          </Badge>
        </template>
        <template #cell-status="{ row }">
          <div class="flex items-center gap-2">
            <span :class="getStatusColor(row.status)" class="w-2 h-2 rounded-full"></span>
            <Badge :variant="getStatusBadgeVariant(row.status)">
              {{ statusMap[row.status] }}
            </Badge>
          </div>
        </template>
        <template #cell-progress="{ row }">
          <div class="flex items-center gap-2">
            <div class="flex-1 w-24 h-2 bg-secondary-200 rounded-full overflow-hidden">
              <div
                :class="getProgressColorClass(row.progress)"
                class="h-full rounded-full transition-all"
                :style="{ width: row.progress + '%' }"
              ></div>
            </div>
            <span class="text-sm text-secondary-600">{{ row.progress }}%</span>
          </div>
        </template>
        <template #cell-lastRun="{ row }">
          {{ formatDate(row.lastRun) }}
        </template>
        <template #actions="{ row }">
          <div class="flex items-center justify-end gap-2">
            <Button
              v-if="row.status === 'pending' || row.status === 'failed'"
              variant="primary"
              size="sm"
              @click="startTask(row)"
            >
              启动
            </Button>
            <Button
              v-else-if="row.status === 'running'"
              variant="warning"
              size="sm"
              @click="stopTask(row)"
            >
              停止
            </Button>
            <Button variant="ghost" size="sm" @click="viewLogs(row)">日志</Button>
            <Button variant="ghost" size="sm" @click="editTask(row)">编辑</Button>
            <Button variant="danger" size="sm" @click="deleteTask(row)">删除</Button>
          </div>
        </template>
      </Table>
    </Card>

    <!-- 新建/编辑任务模态框 -->
    <Modal
      v-model="showCreateModal"
      :title="editingTask ? '编辑任务' : '新建采集任务'"
      size="lg"
      max-height="80vh"
      :show-default-footer="true"
      :loading="submitting"
      @confirm="handleSubmit"
    >
      <div class="space-y-4">
        <!-- 基础配置 -->
        <FormSection title="基础配置">
          <Input
            v-model="taskForm.name"
            label="任务名称"
            placeholder="请输入任务名称"
            :error="formErrors.name"
            required
          />
          <Select
            v-model="taskForm.type"
            label="采集类型"
            :options="typeOptions"
            placeholder="请选择采集类型"
            :error="formErrors.type"
            required
            @change="onTypeChange"
          />
          <Textarea
            v-model="taskForm.description"
            label="任务描述"
            placeholder="请输入任务描述"
            :rows="2"
          />
        </FormSection>

        <!-- 根据类型显示不同的配置 -->
        <FormSection v-if="taskForm.type === 'stock'" title="股票爬虫配置">
          <div class="space-y-3">
            <div class="grid grid-cols-2 gap-3">
              <Input
                v-model="stockForm.symbols"
                label="股票代码"
                placeholder="多个代码用逗号分隔，如：000001,600519"
                :default-value="getDefaultStockConfig().symbols"
              />
              <Select
                v-model="stockForm.market"
                label="市场类型"
                :options="marketOptions"
                :default-value="getDefaultStockConfig().market"
              />
            </div>
            <div class="grid grid-cols-2 gap-3">
              <Select
                v-model="stockForm.fields"
                label="采集字段"
                :options="stockFieldOptions"
                :default-value="getDefaultStockConfig().fields"
                multiple
              />
              <Select
                v-model="stockForm.frequency"
                label="采集频率"
                :options="frequencyOptions"
                :default-value="getDefaultStockConfig().frequency"
              />
            </div>
            <div class="grid grid-cols-2 gap-3">
              <Input
                v-model.number="stockForm.start_time"
                label="开始时间"
                type="time"
                :default-value="getDefaultStockConfig().start_time"
              />
              <Input
                v-model.number="stockForm.end_time"
                label="结束时间"
                type="time"
                :default-value="getDefaultStockConfig().end_time"
              />
            </div>
          </div>
        </FormSection>

        <FormSection v-if="taskForm.type === 'weather'" title="气象爬虫配置">
          <div class="space-y-3">
            <div class="grid grid-cols-2 gap-3">
              <Input
                v-model="weatherForm.cities"
                label="城市列表"
                placeholder="多个城市用逗号分隔，如：北京，上海，广州"
                :default-value="getDefaultWeatherConfig().cities"
              />
              <Select
                v-model="weatherForm.data_type"
                label="数据类型"
                :options="weatherDataTypeOptions"
                :default-value="getDefaultWeatherConfig().data_type"
              />
            </div>
            <div class="grid grid-cols-2 gap-3">
              <Select
                v-model="weatherForm.fields"
                label="采集字段"
                :options="weatherFieldOptions"
                :default-value="getDefaultWeatherConfig().fields"
                multiple
              />
              <Select
                v-model="weatherForm.frequency"
                label="采集频率"
                :options="weatherFrequencyOptions"
                :default-value="getDefaultWeatherConfig().frequency"
              />
            </div>
          </div>
        </FormSection>

        <FormSection v-if="taskForm.type === 'consumption'" title="消费数据爬虫配置">
          <div class="space-y-3">
            <div class="grid grid-cols-2 gap-3">
              <Select
                v-model="consumptionForm.indicator_type"
                label="指标类型"
                :options="indicatorTypeOptions"
                :default-value="getDefaultConsumptionConfig().indicator_type"
              />
              <Select
                v-model="consumptionForm.region"
                label="区域范围"
                :options="regionOptions"
                :default-value="getDefaultConsumptionConfig().region"
              />
            </div>
            <div class="grid grid-cols-2 gap-3">
              <Input
                v-model="consumptionForm.start_year"
                label="开始年份"
                type="number"
                :default-value="getDefaultConsumptionConfig().start_year"
              />
              <Input
                v-model="consumptionForm.end_year"
                label="结束年份"
                type="number"
                :default-value="getDefaultConsumptionConfig().end_year"
              />
            </div>
          </div>
        </FormSection>

        <FormSection v-if="taskForm.type === 'fortune'" title="算命数据爬虫配置">
          <div class="space-y-3">
            <div class="grid grid-cols-2 gap-3">
              <Select
                v-model="fortuneForm.category"
                label="数据分类"
                :options="fortuneCategoryOptions"
                :default-value="getDefaultFortuneConfig().category"
              />
              <Input
                v-model="fortuneForm.keywords"
                label="关键词"
                placeholder="多个关键词用逗号分隔"
                :default-value="getDefaultFortuneConfig().keywords"
              />
            </div>
            <div class="grid grid-cols-2 gap-3">
              <Select
                v-model="fortuneForm.fields"
                label="采集字段"
                :options="fortuneFieldOptions"
                :default-value="getDefaultFortuneConfig().fields"
                multiple
              />
            </div>
          </div>
        </FormSection>

        <!-- 金融数据爬虫配置 -->
        <FormSection v-if="taskForm.type === 'finance' || taskForm.type === 'finance_akshare' || taskForm.type === 'finance_ai'" title="金融数据爬虫配置">
          <div class="space-y-3">
            <div class="grid grid-cols-2 gap-3">
              <Select
                v-model="financeForm.data_type"
                label="数据类型"
                :options="financeDataTypeOptions"
                :default-value="getDefaultFinanceConfig().data_type"
              />
              <Select
                v-model="financeForm.market"
                label="市场类型"
                :options="financeMarketOptions"
                :default-value="getDefaultFinanceConfig().market"
              />
            </div>
            <Input
              v-model="financeForm.symbols"
              label="股票代码"
              placeholder="多个代码用逗号分隔，如：000001,600519"
              :default-value="getDefaultFinanceConfig().symbols"
            />
            <div v-if="financeForm.data_type === 'history'" class="grid grid-cols-2 gap-3">
              <Input
                v-model="financeForm.start_date"
                label="开始日期"
                type="date"
                :default-value="getDefaultFinanceConfig().start_date"
              />
              <Input
                v-model="financeForm.end_date"
                label="结束日期"
                type="date"
                :default-value="getDefaultFinanceConfig().end_date"
              />
            </div>
            <div v-if="financeForm.data_type === 'history'" class="grid grid-cols-2 gap-3">
              <Select
                v-model="financeForm.period"
                label="周期"
                :options="financePeriodOptions"
                :default-value="getDefaultFinanceConfig().period"
              />
              <Select
                v-model="financeForm.adjust"
                label="复权类型"
                :options="financeAdjustOptions"
                :default-value="getDefaultFinanceConfig().adjust"
              />
            </div>
          </div>
        </FormSection>

        <FormSection v-if="taskForm.type === 'finance_ai'" title="AI 智能提取配置">
          <div class="space-y-3">
            <div class="grid grid-cols-2 gap-3">
              <Select
                v-model="financeForm.llm_provider"
                label="LLM 提供商"
                :options="llmProviderOptions"
                :default-value="getDefaultFinanceConfig().llm_provider"
              />
              <Input
                v-model="financeForm.llm_model"
                label="模型名称"
                :default-value="getDefaultFinanceConfig().llm_model"
              />
            </div>
            <div class="grid grid-cols-2 gap-3">
              <Input
                v-model.number="financeForm.start_year"
                label="开始年份"
                type="number"
                :default-value="getDefaultFinanceConfig().start_year"
              />
              <Input
                v-model.number="financeForm.end_year"
                label="结束年份"
                type="number"
                :default-value="getDefaultFinanceConfig().end_year"
              />
            </div>
            <Select
              v-model="financeForm.indicators"
              label="财务指标"
              :options="financialIndicatorOptions"
              :default-value="getDefaultFinanceConfig().indicators"
              multiple
            />
          </div>
        </FormSection>

        <!-- 通用配置 -->
        <FormSection title="调度配置">
          <div class="grid grid-cols-2 gap-3">
            <Select
              v-model="taskForm.schedule"
              label="执行频率"
              :options="scheduleOptions"
              :default-value="'manual'"
            />
            <Input
              v-model.number="taskForm.timeout"
              label="超时时间 (秒)"
              type="number"
              :default-value="300"
            />
          </div>
        </FormSection>

        <!-- 高级配置 -->
        <FormSection title="高级配置">
          <div class="grid grid-cols-2 gap-3">
            <Input
              v-model.number="taskForm.retry_times"
              label="重试次数"
              type="number"
              :min="0"
              :max="5"
              :default-value="3"
            />
            <Input
              v-model.number="taskForm.cache_ttl"
              label="缓存时间 (秒)"
              type="number"
              :default-value="3600"
            />
          </div>
        </FormSection>
      </div>
    </Modal>

    <!-- 日志模态框 -->
    <Modal
      v-model="showLogModal"
      title="任务日志"
      size="xl"
    >
      <div v-if="currentTask" class="space-y-4">
        <div class="flex items-center justify-between">
          <h4 class="font-semibold">{{ currentTask.name }}</h4>
          <Button variant="outline" size="sm" @click="exportLogs">
            <svg class="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            导出日志
          </Button>
        </div>
        <div class="bg-secondary-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-96 overflow-auto">
          <div v-for="(line, index) in logLines" :key="index" class="mb-1">
            {{ line }}
          </div>
        </div>
      </div>
    </Modal>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import Textarea from '@/components/ui/Textarea.vue'
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Badge from '@/components/ui/Badge.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import SearchBar from '@/components/ui/SearchBar.vue'
import StatCard from '@/components/ui/StatCard.vue'

const loading = ref(false)
const submitting = ref(false)
const showCreateModal = ref(false)
const showLogModal = ref(false)
const searchQuery = ref('')
const editingTask = ref(null)

const typeMap = {
  stock: '股票数据',
  weather: '气象数据',
  fortune: '算命数据',
  consumption: '消费数据',
  finance: '金融数据',
  finance_akshare: 'AKShare 金融',
  finance_ai: 'AI 金融',
  general: '通用采集'
}

const statusMap = {
  pending: '待执行',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  stopped: '已停止'
}

const columns = [
  { key: 'name', label: '任务名称' },
  { key: 'type', label: '类型' },
  { key: 'status', label: '状态' },
  { key: 'progress', label: '进度' },
  { key: 'lastRun', label: '上次执行' },
  { key: 'nextRun', label: '下次执行' }
]

const tasks = ref([
  {
    id: 1,
    name: 'A 股实时行情采集',
    type: 'stock',
    status: 'running',
    progress: 65,
    schedule: '5min',
    lastRun: '2024-01-15 10:00:00',
    nextRun: '2024-01-15 11:00:00',
    description: '采集 A 股实时行情数据（新浪财经 + 聚宽 API）',
    timeout: 300,
    retry_times: 3,
    cache_ttl: 60,
    config: {
      // 基础配置
      symbols: '000001,600519,300750',
      market: 'cn',
      fields: ['ts_code', 'name', 'open', 'high', 'low', 'close', 'volume', 'amount'],
      frequency: '5min',
      start_time: '09:30',
      end_time: '15:00',
      
      // 数据源配置
      data_sources: [
        {
          name: '新浪财经',
          url: 'http://hq.sinajs.cn/list={symbol}',
          method: 'GET',
          auth_type: 'none',  // 无需认证
          headers: {
            'Referer': 'https://finance.sina.com.cn/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
          },
          timeout: 10,
          cache_ttl: 60,
          enabled: true
        },
        {
          name: '聚宽数据 API',
          url: 'https://api.joinquant.com/data/stocks',
          method: 'GET',
          auth_type: 'api_key',
          api_config: {
            base_url: 'https://api.joinquant.com',
            api_version: 'v1',
            api_key: 'YOUR_JQ_API_KEY',  // 需要替换为真实的 API Key
            token_type: 'Bearer'
          },
          headers: {
            'Authorization': 'Bearer {api_key}',
            'X-API-Key': '{api_key}'
          },
          timeout: 15,
          cache_ttl: 300,
          enabled: false  // 默认禁用，需要配置 API Key 后启用
        }
      ]
    }
  },
  {
    id: 2,
    name: '全国气象数据采集',
    type: 'weather',
    status: 'completed',
    progress: 100,
    schedule: 'hourly',
    lastRun: '2024-01-15 06:00:00',
    nextRun: '2024-01-15 07:00:00',
    description: '采集全国主要城市气象数据（和风天气 API）',
    timeout: 300,
    retry_times: 3,
    cache_ttl: 1800,
    config: {
      // 基础配置
      cities: '北京，上海，广州，深圳，杭州',
      city_codes: {
        '北京': '101010100',
        '上海': '101020100',
        '广州': '101280101',
        '深圳': '101280601',
        '杭州': '101210101'
      },
      data_type: 'realtime',
      fields: ['temperature', 'humidity', 'weather', 'wind_direction', 'wind_power'],
      frequency: 'hourly',
      
      // 数据源配置
      data_sources: [
        {
          name: '和风天气 API',
          url: 'https://devapi.qweather.com/v7/weather/now',
          method: 'GET',
          auth_type: 'api_key',
          api_config: {
            base_url: 'https://devapi.qweather.com',
            api_version: 'v7',
            api_key: 'YOUR_QWEATHER_KEY',  // 需要替换为真实的 API Key
            token_type: 'Bearer'
          },
          params: {
            'key': '{api_key}',
            'location': '{location_id}'
          },
          timeout: 10,
          cache_ttl: 1800,
          enabled: true  // 需要配置真实的 API Key
        },
        {
          name: '中国天气网',
          url: 'http://www.weather.com.cn/weather/{city_code}.shtml',
          method: 'GET',
          auth_type: 'none',  // 无需认证
          timeout: 15,
          cache_ttl: 1800,
          enabled: true
        }
      ]
    }
  },
  {
    id: 3,
    name: '周易数据采集',
    type: 'fortune',
    status: 'pending',
    progress: 0,
    schedule: 'daily',
    lastRun: '2024-01-14 02:00:00',
    nextRun: '2024-01-15 02:00:00',
    description: '采集周易八卦相关数据（百度百科）',
    timeout: 300,
    retry_times: 3,
    cache_ttl: 86400,
    config: {
      // 基础配置
      category: 'zhouyi',
      keywords: '周易，八卦，六十四卦，乾卦，坤卦',
      fields: ['title', 'content', 'explanation', 'source'],
      
      // 数据源配置
      data_sources: [
        {
          name: '百度百科',
          url: 'https://baike.baidu.com/item/{keyword}',
          method: 'GET',
          auth_type: 'none',  // 无需认证
          headers: {
            'Referer': 'https://baike.baidu.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'text/html,application/xhtml+xml'
          },
          timeout: 15,
          cache_ttl: 86400,
          enabled: true
        }
      ]
    }
  },
  {
    id: 4,
    name: '宏观经济数据采集',
    type: 'consumption',
    status: 'failed',
    progress: 45,
    schedule: 'daily',
    lastRun: '2024-01-15 08:00:00',
    nextRun: '-',
    description: '采集 GDP、CPI 等宏观经济数据（国家统计局）',
    timeout: 300,
    retry_times: 3,
    cache_ttl: 86400,
    config: {
      // 基础配置
      indicator_type: 'gdp',
      region: 'national',
      start_year: 2014,
      end_year: 2024,
      indicators: ['gdp', 'cpi', 'ppi', 'pmi'],
      
      // 数据源配置
      data_sources: [
        {
          name: '国家统计局',
          url: 'https://data.stats.gov.cn/easyquery.htm',
          method: 'GET',
          auth_type: 'none',  // 无需认证
          params: {
            'cn': 'C01',
            'zb': '{indicator_code}',
            'sj': '{year}'
          },
          timeout: 30,
          cache_ttl: 86400,
          enabled: true
        }
      ]
    }
  },
  {
    id: 5,
    name: '通用网页采集',
    type: 'general',
    status: 'stopped',
    progress: 30,
    schedule: 'manual',
    lastRun: '2024-01-14 15:00:00',
    nextRun: '-',
    description: '通用网页数据采集（支持自定义配置）',
    timeout: 300,
    retry_times: 3,
    cache_ttl: 3600,
    config: {
      // 基础配置
      urls: [],
      fields: [],
      
      // 数据源配置
      data_sources: [],
      
      // 认证配置（可选）
      auth_config: {
        auth_type: 'none',  // none, login, api_key, token, oauth2
        login_url: '',
        username: '',
        password: '',
        api_key: '',
        access_token: '',
        token_url: ''
      }
    }
  },
  {
    id: 6,
    name: 'A 股智能金融数据采集',
    type: 'finance',
    status: 'pending',
    progress: 0,
    schedule: '5min',
    lastRun: '-',
    nextRun: '2024-01-15 11:00:00',
    description: '基于 AKShare 和 ScrapeGraphAI 的智能金融数据采集',
    timeout: 300,
    retry_times: 3,
    cache_ttl: 60,
    config: {
      // 基础配置
      data_type: 'realtime_quote',  // realtime_quote/history/financial/stock_basic
      market: 'A 股',
      symbols: '000001,600519,300750',
      
      // 数据源配置
      data_sources: [
        {
          name: 'AKShare 实时行情',
          source_type: 'akshare',
          method: 'GET',
          timeout: 10,
          cache_ttl: 60,
          enabled: true,
          description: '使用 AKShare 采集 A 股实时行情'
        },
        {
          name: '新浪财经（备用）',
          source_type: 'web',
          url: 'http://hq.sinajs.cn/list={symbol}',
          method: 'GET',
          auth_type: 'none',
          headers: {
            'Referer': 'https://finance.sina.com.cn/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
          },
          timeout: 10,
          cache_ttl: 60,
          enabled: false,
          description: '新浪财经备用数据源'
        }
      ],
      
      // 股票配置
      stock_config: {
        data_type: 'realtime_quote',
        market: 'A 股',
        symbols: ['000001', '600519', '300750'],
        period: 'daily',
        adjust: 'qfq',
        fields: ['ts_code', 'name', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']
      }
    }
  },
  {
    id: 7,
    name: '财务指标智能采集',
    type: 'finance_ai',
    status: 'pending',
    progress: 0,
    schedule: 'daily',
    lastRun: '-',
    nextRun: '2024-01-16 02:00:00',
    description: '使用 AI 提取上市公司财务指标数据',
    timeout: 600,
    retry_times: 3,
    cache_ttl: 86400,
    config: {
      // 基础配置
      data_type: 'financial',
      symbols: '000001,600519',
      
      // 数据源配置
      data_sources: [
        {
          name: 'AKShare 财务指标',
          source_type: 'akshare',
          method: 'GET',
          timeout: 30,
          cache_ttl: 86400,
          enabled: true,
          description: '使用 AKShare 采集财务指标数据'
        },
        {
          name: '巨潮资讯网（AI 提取）',
          source_type: 'web',
          url: 'http://www.cninfo.com.cn/',
          method: 'GET',
          auth_type: 'none',
          timeout: 60,
          cache_ttl: 86400,
          enabled: false,
          description: '使用 AI 从巨潮资讯网提取财务数据'
        }
      ],
      
      // 财务指标配置
      indicator_config: {
        ts_code: '',
        start_year: 2020,
        end_year: 2024,
        indicators: ['eps', 'revenue', 'net_profit', 'roe', 'gross_margin']
      },
      
      // LLM 配置
      llm_config: {
        provider: 'ollama',
        model: 'llama3.2',
        temperature: 0.0
      }
    }
  }
])

const taskForm = reactive({
  name: '',
  type: 'general',
  schedule: 'manual',
  description: '',
  timeout: 300,
  retry_times: 3,
  cache_ttl: 3600
})

// 各类型爬虫表单
const stockForm = reactive({})
const weatherForm = reactive({})
const consumptionForm = reactive({})
const fortuneForm = reactive({})
const financeForm = reactive({})  // 金融数据爬虫表单

const formErrors = reactive({
  name: '',
  type: ''
})

const typeOptions = [
  { value: 'stock', label: '股票数据' },
  { value: 'weather', label: '气象数据' },
  { value: 'fortune', label: '算命数据' },
  { value: 'consumption', label: '消费数据' },
  { value: 'finance', label: '金融数据' },
  { value: 'finance_akshare', label: 'AKShare 金融' },
  { value: 'finance_ai', label: 'AI 金融' },
  { value: 'general', label: '通用采集' }
]

const scheduleOptions = [
  { value: 'manual', label: '手动执行' },
  { value: 'hourly', label: '每小时' },
  { value: 'daily', label: '每天' },
  { value: 'weekly', label: '每周' },
  { value: 'monthly', label: '每月' }
]

// 股票市场选项
const marketOptions = [
  { value: 'cn', label: 'A 股' },
  { value: 'hk', label: '港股' },
  { value: 'us', label: '美股' },
  { value: 'all', label: '全部市场' }
]

// 股票采集字段选项
const stockFieldOptions = [
  { value: 'ts_code', label: '股票代码' },
  { value: 'name', label: '股票名称' },
  { value: 'open', label: '开盘价' },
  { value: 'high', label: '最高价' },
  { value: 'low', label: '最低价' },
  { value: 'close', label: '收盘价' },
  { value: 'volume', label: '成交量' },
  { value: 'amount', label: '成交额' }
]

// 频率选项
const frequencyOptions = [
  { value: 'realtime', label: '实时' },
  { value: '5min', label: '5 分钟' },
  { value: '15min', label: '15 分钟' },
  { value: '60min', label: '60 分钟' },
  { value: 'daily', label: '每日' }
]

// 气象数据类型选项
const weatherDataTypeOptions = [
  { value: 'realtime', label: '实时天气' },
  { value: 'forecast', label: '天气预报' },
  { value: 'history', label: '历史数据' },
  { value: 'all', label: '全部类型' }
]

// 气象采集字段选项
const weatherFieldOptions = [
  { value: 'temperature', label: '温度' },
  { value: 'humidity', label: '湿度' },
  { value: 'weather', label: '天气状况' },
  { value: 'wind_direction', label: '风向' },
  { value: 'wind_power', label: '风力' },
  { value: 'precipitation', label: '降水量' }
]

// 气象频率选项
const weatherFrequencyOptions = [
  { value: 'hourly', label: '每小时' },
  { value: '3hours', label: '3 小时' },
  { value: '6hours', label: '6 小时' },
  { value: 'daily', label: '每日' }
]

// 宏观经济指标类型选项
const indicatorTypeOptions = [
  { value: 'gdp', label: 'GDP' },
  { value: 'cpi', label: 'CPI' },
  { value: 'ppi', label: 'PPI' },
  { value: 'pmi', label: 'PMI' },
  { value: 'all', label: '全部指标' }
]

// 区域选项
const regionOptions = [
  { value: 'national', label: '全国' },
  { value: 'province', label: '省份' },
  { value: 'city', label: '城市' },
  { value: 'all', label: '全部区域' }
]

// 算命数据分类选项
const fortuneCategoryOptions = [
  { value: 'zhouyi', label: '周易' },
  { value: 'bazi', label: '八字' },
  { value: 'mianxiang', label: '面相' },
  { value: 'fengshui', label: '风水' },
  { value: 'all', label: '全部分类' }
]

// 算命采集字段选项
const fortuneFieldOptions = [
  { value: 'title', label: '标题' },
  { value: 'content', label: '内容' },
  { value: 'explanation', label: '解释' },
  { value: 'source', label: '出处' }
]

// 金融数据类型选项
const financeDataTypeOptions = [
  { value: 'realtime_quote', label: '实时行情' },
  { value: 'history', label: '历史行情' },
  { value: 'financial', label: '财务指标' },
  { value: 'stock_basic', label: '股票基础信息' }
]

// 金融市场选项
const financeMarketOptions = [
  { value: 'A 股', label: 'A 股' },
  { value: '港股', label: '港股' },
  { value: '美股', label: '美股' }
]

// 金融周期选项
const financePeriodOptions = [
  { value: 'daily', label: '日线' },
  { value: 'weekly', label: '周线' },
  { value: 'monthly', label: '月线' }
]

// 复权类型选项
const financeAdjustOptions = [
  { value: 'qfq', label: '前复权' },
  { value: 'hfq', label: '后复权' },
  { value: 'no', label: '不复权' }
]

// LLM 提供商选项
const llmProviderOptions = [
  { value: 'ollama', label: 'Ollama' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'groq', label: 'Groq' }
]

// 财务指标选项
const financialIndicatorOptions = [
  { value: 'eps', label: '每股收益 (EPS)' },
  { value: 'revenue', label: '营业收入' },
  { value: 'net_profit', label: '净利润' },
  { value: 'roe', label: '净资产收益率 (ROE)' },
  { value: 'gross_margin', label: '毛利率' },
  { value: 'pe_ratio', label: '市盈率' },
  { value: 'pb_ratio', label: '市净率' }
]

const currentTask = ref(null)

const filteredTasks = computed(() => {
  if (!searchQuery.value) return tasks.value
  return tasks.value.filter(task =>
    task.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
    typeMap[task.type].includes(searchQuery.value)
  )
})

const runningCount = computed(() => tasks.value.filter(t => t.status === 'running').length)
const completedCount = computed(() => tasks.value.filter(t => t.status === 'completed').length)
const failedCount = computed(() => tasks.value.filter(t => t.status === 'failed').length)

const logLines = ref([
  '[2024-01-15 10:00:00] 任务启动...',
  '[2024-01-15 10:00:01] 连接目标服务器...',
  '[2024-01-15 10:00:02] 连接成功，开始采集...',
  '[2024-01-15 10:00:15] 已采集 100 条数据...',
  '[2024-01-15 10:00:30] 已采集 200 条数据...',
  '[2024-01-15 10:00:45] 数据采集完成，共 350 条',
  '[2024-01-15 10:00:46] 数据清洗中...',
  '[2024-01-15 10:00:50] 数据入库成功',
  '[2024-01-15 10:00:51] 任务完成'
])

const getTypeBadgeVariant = (type) => {
  const variants = {
    stock: 'primary',
    weather: 'info',
    fortune: 'warning',
    consumption: 'success',
    finance: 'primary',
    finance_akshare: 'primary',
    finance_ai: 'success',
    general: 'secondary'
  }
  return variants[type] || 'secondary'
}

const getStatusBadgeVariant = (status) => {
  const variants = {
    pending: 'secondary',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    stopped: 'secondary'
  }
  return variants[status] || 'secondary'
}

const getStatusColor = (status) => {
  const colors = {
    pending: 'bg-secondary-400',
    running: 'bg-warning-500',
    completed: 'bg-success-500',
    failed: 'bg-danger-500',
    stopped: 'bg-secondary-500'
  }
  return colors[status] || 'bg-secondary-400'
}

const getProgressColorClass = (progress) => {
  if (progress < 50) return 'bg-warning-500'
  if (progress < 100) return 'bg-primary-500'
  return 'bg-success-500'
}

const formatDate = (date) => {
  if (!date || date === '-') return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const handleSearch = () => {}

const startTask = async (task) => {
  try {
    // 更新状态
    task.status = 'running'
    task.progress = 0
    
    // 调用后端 API 启动爬虫
    const response = await api.post('/crawler-task/start', {
      task_id: task.id,
      crawler_type: task.type,
      config: task.config,
      timeout: task.timeout,
      retry_times: task.retry_times,
      cache_ttl: task.cache_ttl
    })
    
    if (response.success) {
      // 启动成功，模拟进度更新
      const progressInterval = setInterval(() => {
        if (task.progress < 90) {
          task.progress += 10
        }
      }, 1000)
      
      // 模拟任务完成
      setTimeout(() => {
        clearInterval(progressInterval)
        task.progress = 100
        task.status = 'completed'
        task.lastRun = new Date().toLocaleString('zh-CN')
      }, 10000)
    } else {
      task.status = 'failed'
      alert('启动失败：' + (response.message || '未知错误'))
    }
  } catch (error) {
    console.error('启动任务失败:', error)
    task.status = 'failed'
    alert('启动失败：' + error.message)
  }
}

const stopTask = async (task) => {
  task.status = 'stopped'
}

const viewLogs = (task) => {
  currentTask.value = task
  showLogModal.value = true
}

const deleteTask = async (task) => {
  if (confirm(`确定要删除任务 "${task.name}" 吗？`)) {
    tasks.value = tasks.value.filter(t => t.id !== task.id)
  }
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 根据类型保存对应的配置
    let config = {}
    if (taskForm.type === 'stock') {
      config = { ...stockForm }
    } else if (taskForm.type === 'weather') {
      config = { ...weatherForm }
    } else if (taskForm.type === 'consumption') {
      config = { ...consumptionForm }
    } else if (taskForm.type === 'fortune') {
      config = { ...fortuneForm }
    } else if (taskForm.type === 'finance' || taskForm.type === 'finance_akshare' || taskForm.type === 'finance_ai') {
      config = { ...financeForm }
    }
    
    const taskData = {
      ...taskForm,
      config,
      status: editingTask.value ? editingTask.value.status : 'pending',
      progress: editingTask.value ? editingTask.value.progress : 0,
      lastRun: editingTask.value ? editingTask.value.lastRun : '-',
      nextRun: editingTask.value ? editingTask.value.nextRun : '-'
    }
    
    if (editingTask.value) {
      Object.assign(editingTask.value, taskData)
    } else {
      tasks.value.unshift({
        id: Date.now(),
        ...taskData
      })
    }
    showCreateModal.value = false
    resetForm()
  } finally {
    submitting.value = false
  }
}

// 获取各类型默认配置
const getDefaultStockConfig = () => ({
  symbols: '000001,600519,300750',
  market: 'cn',
  fields: ['ts_code', 'name', 'open', 'high', 'low', 'close', 'volume', 'amount'],
  frequency: '5min',
  start_time: '09:30',
  end_time: '15:00'
})

const getDefaultWeatherConfig = () => ({
  cities: '北京，上海，广州，深圳',
  data_type: 'realtime',
  fields: ['temperature', 'humidity', 'weather', 'wind_direction', 'wind_power'],
  frequency: 'hourly'
})

const getDefaultConsumptionConfig = () => ({
  indicator_type: 'gdp',
  region: 'national',
  start_year: new Date().getFullYear() - 10,
  end_year: new Date().getFullYear()
})

const getDefaultFortuneConfig = () => ({
  category: 'zhouyi',
  keywords: '周易，八卦，六十四卦',
  fields: ['title', 'content', 'explanation', 'source']
})

// 获取金融数据默认配置
const getDefaultFinanceConfig = () => ({
  data_type: 'realtime_quote',
  market: 'A 股',
  symbols: '000001,600519,300750',
  start_date: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
  end_date: new Date().toISOString().split('T')[0],
  period: 'daily',
  adjust: 'qfq',
  llm_provider: 'ollama',
  llm_model: 'llama3.2',
  start_year: 2020,
  end_year: new Date().getFullYear(),
  indicators: ['eps', 'revenue', 'net_profit', 'roe']
})

// 类型切换处理
const onTypeChange = () => {
  // 根据选择的类型，加载默认配置
  if (taskForm.type === 'stock') {
    Object.assign(stockForm, getDefaultStockConfig())
  } else if (taskForm.type === 'weather') {
    Object.assign(weatherForm, getDefaultWeatherConfig())
  } else if (taskForm.type === 'consumption') {
    Object.assign(consumptionForm, getDefaultConsumptionConfig())
  } else if (taskForm.type === 'fortune') {
    Object.assign(fortuneForm, getDefaultFortuneConfig())
  } else if (taskForm.type === 'finance' || taskForm.type === 'finance_akshare' || taskForm.type === 'finance_ai') {
    Object.assign(financeForm, getDefaultFinanceConfig())
  }
}

const resetForm = () => {
  editingTask.value = null
  Object.assign(taskForm, {
    name: '',
    type: 'general',
    schedule: 'manual',
    description: '',
    timeout: 300,
    retry_times: 3,
    cache_ttl: 3600
  })
  // 重置各类型表单
  Object.assign(stockForm, {})
  Object.assign(weatherForm, {})
  Object.assign(consumptionForm, {})
  Object.assign(fortuneForm, {})
  Object.assign(financeForm, {})
  Object.assign(formErrors, {
    name: '',
    type: ''
  })
}

const editTask = (task) => {
  editingTask.value = task
  Object.assign(taskForm, {
    name: task.name,
    type: task.type,
    schedule: task.schedule || 'manual',
    description: task.description,
    timeout: task.timeout || 300,
    retry_times: task.retry_times || 3,
    cache_ttl: task.cache_ttl || 3600
  })
  
  // 加载对应类型的配置
  if (task.type === 'stock' && task.config) {
    Object.assign(stockForm, task.config)
  } else if (task.type === 'weather' && task.config) {
    Object.assign(weatherForm, task.config)
  } else if (task.type === 'consumption' && task.config) {
    Object.assign(consumptionForm, task.config)
  } else if (task.type === 'fortune' && task.config) {
    Object.assign(fortuneForm, task.config)
  }
  
  showCreateModal.value = true
}

const exportLogs = () => {
  alert('正在导出日志...')
}

onMounted(() => {
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 500)
})
</script>
