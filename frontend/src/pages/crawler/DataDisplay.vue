<template>
  <div class="space-y-6">
    <PageHeader title="爬虫数据展示" subtitle="实时查看采集到的数据">
      <template #actions>
        <Button variant="outline" @click="refreshData">
          <svg class="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          刷新数据
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '爬虫采集', to: '/crawler' },
      { label: '数据展示' }
    ]" />

    <!-- 数据概览 -->
    <Card>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard title="股票数据" :value="stats.stocks.stock_count.toString()" icon="📈" :subtitle="`行情记录：${stats.stocks.quote_count}`" />
        <StatCard title="气象数据" :value="stats.weather.station_count.toString()" icon="🌤️" :subtitle="`气象记录：${stats.weather.record_count}`" />
        <StatCard title="GDP 数据" :value="stats.consumption.gdp_count.toString()" icon="💰" :subtitle="`人口记录：${stats.consumption.population_count}`" />
        <StatCard title="总记录数" :value="stats.total_records.toString()" icon="📊" trend="+2.5%" trend-positive />
      </div>
    </Card>

    <!-- 选项卡切换 -->
    <Card>
      <Tabs v-model="activeTab" :tabs="tabOptions" />

      <div class="mt-6">
        <!-- 股票数据 -->
        <template v-if="activeTab === 'stocks'">
          <div class="flex items-center justify-between mb-4">
            <h3 class="font-semibold text-lg">股票数据</h3>
            <div class="flex items-center gap-2">
              <Input
                v-model="stockSearch"
                placeholder="搜索股票代码或名称"
                class="w-64"
                @input="filterStocks"
              />
            </div>
          </div>

          <Table :columns="stockColumns" :data="filteredStocks" :loading="loading">
            <template #cell-code="{ row }">
              <span class="font-mono font-medium">{{ row.ts_code }}</span>
            </template>
            <template #cell-name="{ row }">
              <div>
                <div class="font-medium">{{ row.name }}</div>
                <div class="text-xs text-secondary-500">{{ row.industry }}</div>
              </div>
            </template>
            <template #cell-price="{ row }">
              <span :class="row.change_percent >= 0 ? 'text-red-500' : 'text-green-500'" class="font-semibold">
                ¥{{ row.close }}
              </span>
            </template>
            <template #cell-change="{ row }">
              <span :class="row.change_percent >= 0 ? 'text-red-500' : 'text-green-500'">
                {{ row.change_percent >= 0 ? '+' : '' }}{{ row.change_percent }}%
              </span>
            </template>
            <template #actions="{ row }">
              <Button variant="outline" size="sm" @click="viewStockDetail(row)">详情</Button>
            </template>
          </Table>
        </template>

        <!-- 气象数据 -->
        <template v-if="activeTab === 'weather'">
          <div class="flex items-center justify-between mb-4">
            <h3 class="font-semibold text-lg">气象站点数据</h3>
            <Input
              v-model="weatherSearch"
              placeholder="搜索站点名称"
              class="w-64"
              @input="filterWeather"
            />
          </div>

          <Table :columns="weatherColumns" :data="filteredWeatherStations" :loading="loading">
            <template #cell-station_id="{ row }">
              <span class="font-mono">{{ row.station_id }}</span>
            </template>
            <template #cell-location="{ row }">
              <div>
                <div class="font-medium">{{ row.station_name }}</div>
                <div class="text-xs text-secondary-500">{{ row.province }} - {{ row.city }}</div>
              </div>
            </template>
            <template #cell-last_update="{ row }">
              {{ formatDateTime(row.last_update) }}
            </template>
            <template #actions="{ row }">
              <Button variant="outline" size="sm" @click="viewWeatherDetail(row)">详情</Button>
            </template>
          </Table>
        </template>

        <!-- 消费数据 -->
        <template v-if="activeTab === 'consumption'">
          <div class="flex items-center justify-between mb-4">
            <h3 class="font-semibold text-lg">宏观经济数据</h3>
            <Select
              v-model="consumptionFilter"
              :options="consumptionOptions"
              class="w-48"
              @change="filterConsumption"
            />
          </div>

          <Table :columns="consumptionColumns" :data="filteredConsumption" :loading="loading">
            <template #cell-indicator_name="{ row }">
              <div>
                <div class="font-medium">{{ row.indicator_name }}</div>
                <div class="text-xs text-secondary-500">{{ row.region_name }}</div>
              </div>
            </template>
            <template #cell-value="{ row }">
              <span class="font-semibold">{{ row.value }}</span>
              <span class="text-xs text-secondary-500 ml-1">{{ row.unit }}</span>
            </template>
            <template #cell-yoy_growth="{ row }">
              <span :class="row.yoy_growth >= 0 ? 'text-red-500' : 'text-green-500'">
                {{ row.yoy_growth >= 0 ? '+' : '' }}{{ row.yoy_growth }}%
              </span>
            </template>
            <template #cell-period="{ row }">
              {{ row.year }}年{{ row.month ? row.month + '月' : '' }}
            </template>
          </Table>
        </template>
      </div>
    </Card>

    <!-- 股票详情模态框 -->
    <Modal v-model="showStockDetail" title="股票详情" size="xl" max-height="80vh">
      <div v-if="currentStock" class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div>
            <div class="text-secondary-500 text-sm">股票代码</div>
            <div class="font-mono font-medium">{{ currentStock.ts_code }}</div>
          </div>
          <div>
            <div class="text-secondary-500 text-sm">股票名称</div>
            <div class="font-medium">{{ currentStock.name }}</div>
          </div>
          <div>
            <div class="text-secondary-500 text-sm">当前价格</div>
            <div class="text-2xl font-bold" :class="currentStock.change_percent >= 0 ? 'text-red-500' : 'text-green-500'">
              ¥{{ currentStock.close }}
            </div>
          </div>
          <div>
            <div class="text-secondary-500 text-sm">涨跌幅</div>
            <div :class="currentStock.change_percent >= 0 ? 'text-red-500' : 'text-green-500'" class="font-semibold">
              {{ currentStock.change_percent >= 0 ? '+' : '' }}{{ currentStock.change_percent }}%
            </div>
          </div>
        </div>

        <!-- K 线图 -->
        <div class="border rounded-lg p-4">
          <h4 class="font-semibold mb-2">价格走势</h4>
          <LineChart
            :data="stockChartData"
            :options="{ responsive: true, maintainAspectRatio: false }"
            height="300"
          />
        </div>
      </div>
    </Modal>

    <!-- 气象详情模态框 -->
    <Modal v-model="showWeatherDetail" title="气象站点详情" size="xl" max-height="80vh">
      <div v-if="currentWeatherStation" class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div>
            <div class="text-secondary-500 text-sm">站点 ID</div>
            <div class="font-mono">{{ currentWeatherStation.station_id }}</div>
          </div>
          <div>
            <div class="text-secondary-500 text-sm">站点名称</div>
            <div class="font-medium">{{ currentWeatherStation.station_name }}</div>
          </div>
          <div>
            <div class="text-secondary-500 text-sm">位置</div>
            <div>{{ currentWeatherStation.province }} - {{ currentWeatherStation.city }}</div>
          </div>
          <div>
            <div class="text-secondary-500 text-sm">海拔</div>
            <div>{{ currentWeatherStation.altitude }}m</div>
          </div>
        </div>

        <!-- 温度曲线图 -->
        <div class="border rounded-lg p-4">
          <h4 class="font-semibold mb-2">温度变化趋势</h4>
          <LineChart
            :data="weatherChartData"
            :options="{ responsive: true, maintainAspectRatio: false }"
            height="300"
          />
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
import Select from '@/components/ui/Select.vue'
import Table from '@/components/ui/Table.vue'
import Modal from '@/components/ui/Modal.vue'
import Tabs from '@/components/ui/Tabs.vue'
import StatCard from '@/components/ui/StatCard.vue'
import LineChart from '@/components/charts/LineChart.vue'
import api from '@/api'

const loading = ref(false)
const activeTab = ref('stocks')
const stockSearch = ref('')
const weatherSearch = ref('')
const consumptionFilter = ref('all')

const showStockDetail = ref(false)
const showWeatherDetail = ref(false)
const currentStock = ref(null)
const currentWeatherStation = ref(null)

const stats = reactive({
  stocks: { stock_count: 0, quote_count: 0 },
  weather: { station_count: 0, record_count: 0 },
  consumption: { gdp_count: 0, population_count: 0, indicator_count: 0 },
  total_records: 0
})

const tabOptions = [
  { value: 'stocks', label: '股票数据' },
  { value: 'weather', label: '气象数据' },
  { value: 'consumption', label: '消费数据' }
]

const stockColumns = [
  { key: 'code', label: '代码' },
  { key: 'name', label: '名称' },
  { key: 'price', label: '当前价' },
  { key: 'change', label: '涨跌幅' },
  { key: 'volume', label: '成交量' },
  { key: 'actions', label: '操作' }
]

const weatherColumns = [
  { key: 'station_id', label: '站点 ID' },
  { key: 'location', label: '位置' },
  { key: 'latitude', label: '纬度' },
  { key: 'longitude', label: '经度' },
  { key: 'last_update', label: '更新时间' },
  { key: 'actions', label: '操作' }
]

const consumptionColumns = [
  { key: 'indicator_name', label: '指标名称' },
  { key: 'period', label: '时期' },
  { key: 'value', label: '数值' },
  { key: 'yoy_growth', label: '同比增长' },
  { key: 'unit', label: '单位' }
]

const stocks = ref([])
const weatherStations = ref([])
const consumptionData = ref([])

const filteredStocks = computed(() => {
  if (!stockSearch.value) return stocks.value
  return stocks.value.filter(s =>
    s.ts_code.includes(stockSearch.value.toUpperCase()) ||
    s.name.includes(stockSearch.value)
  )
})

const filteredWeatherStations = computed(() => {
  if (!weatherSearch.value) return weatherStations.value
  return weatherStations.value.filter(w =>
    w.station_name.includes(weatherSearch.value)
  )
})

const filteredConsumption = computed(() => {
  if (consumptionFilter.value === 'all') return consumptionData.value
  return consumptionData.value.filter(c => c.indicator_type === consumptionFilter.value)
})

const consumptionOptions = [
  { value: 'all', label: '全部指标' },
  { value: 'gdp', label: 'GDP' },
  { value: 'cpi', label: 'CPI' },
  { value: 'ppi', label: 'PPI' },
  { value: 'pmi', label: 'PMI' }
]

const stockChartData = ref({
  labels: [],
  datasets: [{
    label: '收盘价',
    data: [],
    borderColor: 'rgb(59, 130, 246)',
    tension: 0.1
  }]
})

const weatherChartData = ref({
  labels: [],
  datasets: [
    {
      label: '最高温',
      data: [],
      borderColor: 'rgb(239, 68, 68)',
      tension: 0.1
    },
    {
      label: '最低温',
      data: [],
      borderColor: 'rgb(59, 130, 246)',
      tension: 0.1
    }
  ]
})

const filterStocks = () => {}
const filterWeather = () => {}
const filterConsumption = () => {}

const formatDateTime = (date) => {
  if (!date) return '-'
  return new Date(date).toLocaleString('zh-CN')
}

const loadStatistics = async () => {
  try {
    const response = await api.get('/crawler-display/statistics')
    if (response.success) {
      Object.assign(stats, response.data)
    }
  } catch (error) {
    console.error('加载统计数据失败:', error)
  }
}

const loadStocks = async () => {
  try {
    const response = await api.get('/crawler-display/stocks/overview', {
      params: { page: 1, page_size: 100 }
    })
    if (response.success) {
      stocks.value = response.data.stocks || []
    }
  } catch (error) {
    console.error('加载股票数据失败:', error)
  }
}

const loadWeatherStations = async () => {
  try {
    const response = await api.get('/crawler-display/weather/overview', {
      params: { page: 1, page_size: 100 }
    })
    if (response.success) {
      weatherStations.value = response.data.stations || []
    }
  } catch (error) {
    console.error('加载气象站点失败:', error)
  }
}

const loadConsumptionData = async () => {
  try {
    const [gdpRes, indicatorRes] = await Promise.all([
      api.get('/crawler-display/consumption/gdp/overview'),
      api.get('/crawler-display/consumption/economic/chart')
    ])
    
    if (gdpRes.success) {
      consumptionData.value.push(...(gdpRes.data.gdp_data || []))
    }
  } catch (error) {
    console.error('加载消费数据失败:', error)
  }
}

const viewStockDetail = async (stock) => {
  currentStock.value = stock
  showStockDetail.value = true
  
  try {
    const response = await api.get(`/crawler-display/stocks/${stock.ts_code}/chart`)
    if (response.success) {
      const chartData = response.data.chart_data || []
      stockChartData.value = {
        labels: chartData.map(d => d.date),
        datasets: [{
          label: '收盘价',
          data: chartData.map(d => d.close),
          borderColor: 'rgb(59, 130, 246)',
          tension: 0.1
        }]
      }
    }
  } catch (error) {
    console.error('加载股票图表失败:', error)
  }
}

const viewWeatherDetail = async (station) => {
  currentWeatherStation.value = station
  showWeatherDetail.value = true
  
  try {
    const response = await api.get(`/crawler-display/weather/${station.station_id}/chart`)
    if (response.success) {
      const chartData = response.data.chart_data || []
      weatherChartData.value = {
        labels: chartData.map(d => d.date),
        datasets: [
          {
            label: '最高温',
            data: chartData.map(d => d.max_temp),
            borderColor: 'rgb(239, 68, 68)',
            tension: 0.1
          },
          {
            label: '最低温',
            data: chartData.map(d => d.min_temp),
            borderColor: 'rgb(59, 130, 246)',
            tension: 0.1
          }
        ]
      }
    }
  } catch (error) {
    console.error('加载气象图表失败:', error)
  }
}

const refreshData = () => {
  loadStatistics()
  loadStocks()
  loadWeatherStations()
  loadConsumptionData()
}

onMounted(() => {
  loadStatistics()
  loadStocks()
  loadWeatherStations()
  loadConsumptionData()
})
</script>
