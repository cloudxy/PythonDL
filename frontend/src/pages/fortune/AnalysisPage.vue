<template>
  <div class="space-y-6">
    <PageHeader title="综合分析" subtitle="综合运势分析系统">
      <template #actions>
        <Button variant="primary" @click="startAnalysis">
          <svg class="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          开始分析
        </Button>
      </template>
    </PageHeader>

    <Breadcrumb :items="[
      { label: '看相算命' },
      { label: '综合分析' }
    ]" />

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 左侧：输入区域 -->
      <Card class="lg:col-span-1">
        <template #header>
          <h3 class="text-lg font-semibold">分析参数</h3>
        </template>
        <div class="space-y-4">
          <Input v-model="form.name" label="姓名" placeholder="请输入姓名" />
          <Select
            v-model="form.gender"
            label="性别"
            :options="genderOptions"
            placeholder="请选择性别"
          />
          <Input
            v-model="form.birthdate"
            label="出生日期"
            type="date"
            placeholder="请选择出生日期"
          />
          <Input
            v-model="form.birthtime"
            label="出生时间"
            type="time"
            placeholder="请选择出生时间"
          />
          <Select
            v-model="form.zodiac"
            label="生肖"
            :options="zodiacOptions"
            placeholder="请选择生肖"
          />
          <Select
            v-model="form.constellation"
            label="星座"
            :options="constellationOptions"
            placeholder="请选择星座"
          />
          <Button variant="primary" block @click="analyze">
            开始分析
          </Button>
        </div>
      </Card>

      <!-- 右侧：分析结果 -->
      <div class="lg:col-span-2 space-y-6">
        <!-- 总体运势 -->
        <Card>
          <template #header>
            <h3 class="text-lg font-semibold">总体运势</h3>
          </template>
          <div class="flex items-center gap-6">
            <div class="relative w-32 h-32">
              <svg class="w-32 h-32 transform -rotate-90">
                <circle cx="64" cy="64" r="56" stroke="#e5e7eb" stroke-width="16" fill="none" />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="#10b981"
                  stroke-width="16"
                  fill="none"
                  stroke-dasharray="351.86"
                  stroke-dashoffset="351.86 * 0.25"
                  stroke-linecap="round"
                />
              </svg>
              <div class="absolute inset-0 flex items-center justify-center">
                <span class="text-2xl font-bold">75</span>
              </div>
            </div>
            <div class="flex-1">
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <p class="text-sm text-secondary-500">事业运</p>
                  <div class="flex items-center gap-2">
                    <div class="flex-1 h-2 bg-secondary-200 rounded-full overflow-hidden">
                      <div class="h-full bg-primary-500 rounded-full" style="width: 80%"></div>
                    </div>
                    <span class="text-sm font-medium">80</span>
                  </div>
                </div>
                <div>
                  <p class="text-sm text-secondary-500">财运</p>
                  <div class="flex items-center gap-2">
                    <div class="flex-1 h-2 bg-secondary-200 rounded-full overflow-hidden">
                      <div class="h-full bg-warning-500 rounded-full" style="width: 65%"></div>
                    </div>
                    <span class="text-sm font-medium">65</span>
                  </div>
                </div>
                <div>
                  <p class="text-sm text-secondary-500">健康运</p>
                  <div class="flex items-center gap-2">
                    <div class="flex-1 h-2 bg-secondary-200 rounded-full overflow-hidden">
                      <div class="h-full bg-success-500 rounded-full" style="width: 85%"></div>
                    </div>
                    <span class="text-sm font-medium">85</span>
                  </div>
                </div>
                <div>
                  <p class="text-sm text-secondary-500">感情运</p>
                  <div class="flex items-center gap-2">
                    <div class="flex-1 h-2 bg-secondary-200 rounded-full overflow-hidden">
                      <div class="h-full bg-danger-500 rounded-full" style="width: 55%"></div>
                    </div>
                    <span class="text-sm font-medium">55</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        <!-- 八字分析 -->
        <Card>
          <template #header>
            <h3 class="text-lg font-semibold">八字分析</h3>
          </template>
          <div class="grid grid-cols-4 gap-4">
            <div class="text-center p-4 bg-secondary-50 rounded-lg">
              <p class="text-sm text-secondary-500 mb-1">年柱</p>
              <p class="text-xl font-bold text-secondary-900">甲子</p>
            </div>
            <div class="text-center p-4 bg-secondary-50 rounded-lg">
              <p class="text-sm text-secondary-500 mb-1">月柱</p>
              <p class="text-xl font-bold text-secondary-900">乙丑</p>
            </div>
            <div class="text-center p-4 bg-secondary-50 rounded-lg">
              <p class="text-sm text-secondary-500 mb-1">日柱</p>
              <p class="text-xl font-bold text-secondary-900">丙寅</p>
            </div>
            <div class="text-center p-4 bg-secondary-50 rounded-lg">
              <p class="text-sm text-secondary-500 mb-1">时柱</p>
              <p class="text-xl font-bold text-secondary-900">丁卯</p>
            </div>
          </div>
        </Card>

        <!-- 五行分析 -->
        <Card>
          <template #header>
            <h3 class="text-lg font-semibold">五行分析</h3>
          </template>
          <div class="flex items-center gap-4">
            <div class="flex-1 space-y-2">
              <div class="flex items-center gap-2">
                <span class="w-8 text-sm">金</span>
                <div class="flex-1 h-3 bg-secondary-200 rounded-full overflow-hidden">
                  <div class="h-full bg-yellow-500 rounded-full" style="width: 40%"></div>
                </div>
                <span class="text-sm w-8">40%</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-8 text-sm">木</span>
                <div class="flex-1 h-3 bg-secondary-200 rounded-full overflow-hidden">
                  <div class="h-full bg-green-500 rounded-full" style="width: 60%"></div>
                </div>
                <span class="text-sm w-8">60%</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-8 text-sm">水</span>
                <div class="flex-1 h-3 bg-secondary-200 rounded-full overflow-hidden">
                  <div class="h-full bg-blue-500 rounded-full" style="width: 30%"></div>
                </div>
                <span class="text-sm w-8">30%</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-8 text-sm">火</span>
                <div class="flex-1 h-3 bg-secondary-200 rounded-full overflow-hidden">
                  <div class="h-full bg-red-500 rounded-full" style="width: 70%"></div>
                </div>
                <span class="text-sm w-8">70%</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-8 text-sm">土</span>
                <div class="flex-1 h-3 bg-secondary-200 rounded-full overflow-hidden">
                  <div class="h-full bg-orange-500 rounded-full" style="width: 50%"></div>
                </div>
                <span class="text-sm w-8">50%</span>
              </div>
            </div>
          </div>
        </Card>

        <!-- 运势建议 -->
        <Card>
          <template #header>
            <h3 class="text-lg font-semibold">运势建议</h3>
          </template>
          <div class="space-y-3">
            <div class="flex items-start gap-3">
              <span class="text-xl">💼</span>
              <div>
                <p class="font-medium">事业方面</p>
                <p class="text-sm text-secondary-600">近期事业运势较好，适合拓展业务，但需注意细节处理。</p>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <span class="text-xl">💰</span>
              <div>
                <p class="font-medium">财运方面</p>
                <p class="text-sm text-secondary-600">财运平稳，不宜进行高风险投资，建议保守理财。</p>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <span class="text-xl">❤️</span>
              <div>
                <p class="font-medium">感情方面</p>
                <p class="text-sm text-secondary-600">感情运势一般，多与伴侣沟通，避免不必要的误会。</p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import Select from '@/components/ui/Select.vue'

const form = reactive({
  name: '',
  gender: '',
  birthdate: '',
  birthtime: '',
  zodiac: '',
  constellation: ''
})

const genderOptions = [
  { value: 'male', label: '男' },
  { value: 'female', label: '女' }
]

const zodiacOptions = [
  { value: 'rat', label: '鼠' },
  { value: 'ox', label: '牛' },
  { value: 'tiger', label: '虎' },
  { value: 'rabbit', label: '兔' },
  { value: 'dragon', label: '龙' },
  { value: 'snake', label: '蛇' },
  { value: 'horse', label: '马' },
  { value: 'goat', label: '羊' },
  { value: 'monkey', label: '猴' },
  { value: 'rooster', label: '鸡' },
  { value: 'dog', label: '狗' },
  { value: 'pig', label: '猪' }
]

const constellationOptions = [
  { value: 'aries', label: '白羊座' },
  { value: 'taurus', label: '金牛座' },
  { value: 'gemini', label: '双子座' },
  { value: 'cancer', label: '巨蟹座' },
  { value: 'leo', label: '狮子座' },
  { value: 'virgo', label: '处女座' },
  { value: 'libra', label: '天秤座' },
  { value: 'scorpio', label: '天蝎座' },
  { value: 'sagittarius', label: '射手座' },
  { value: 'capricorn', label: '摩羯座' },
  { value: 'aquarius', label: '水瓶座' },
  { value: 'pisces', label: '双鱼座' }
]

const startAnalysis = () => {
  alert('请填写完整的分析参数')
}

const analyze = () => {
  if (!form.name || !form.birthdate) {
    alert('请填写姓名和出生日期')
    return
  }
  alert('分析完成！')
}
</script>
