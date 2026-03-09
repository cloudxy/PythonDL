<template>
  <div class="space-y-6">
    <PageHeader title="星座数据" subtitle="十二星座运势数据管理" />

    <Breadcrumb :items="[{ label: '看相算命' }, { label: '星座数据' }]" />

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      <Card v-for="constellation in constellations" :key="constellation.id">
        <div class="text-center">
          <div class="text-4xl mb-2">{{ constellation.symbol }}</div>
          <h3 class="text-lg font-bold">{{ constellation.name }}</h3>
          <p class="text-sm text-secondary-500">{{ constellation.dateRange }}</p>
          <div class="mt-4 space-y-2">
            <div class="flex justify-between text-sm">
              <span class="text-secondary-500">综合运势</span>
              <Rating :model-value="constellation.overallLuck" readonly />
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-secondary-500">爱情运势</span>
              <Rating :model-value="constellation.loveLuck" readonly />
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-secondary-500">事业运势</span>
              <Rating :model-value="constellation.careerLuck" readonly />
            </div>
          </div>
        </div>
        <template #footer>
          <Button variant="ghost" block @click="editConstellation(constellation)">编辑运势</Button>
        </template>
      </Card>
    </div>

    <Modal v-model="showEditModal" title="编辑星座运势" size="md" :show-default-footer="true" @confirm="handleSave">
      <FormSection v-if="editingConstellation">
        <div class="text-center mb-4">
          <div class="text-4xl">{{ editingConstellation.symbol }}</div>
          <h3 class="text-lg font-bold">{{ editingConstellation.name }}</h3>
        </div>
        <Slider v-model="editForm.overallLuck" label="综合运势" :max="5" />
        <Slider v-model="editForm.loveLuck" label="爱情运势" :max="5" />
        <Slider v-model="editForm.careerLuck" label="事业运势" :max="5" />
        <Slider v-model="editForm.wealthLuck" label="财富运势" :max="5" />
        <Textarea v-model="editForm.dailyHoroscope" label="今日运势" />
        <Textarea v-model="editForm.luckyItem" label="幸运物品" />
      </FormSection>
    </Modal>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import Breadcrumb from '@/components/ui/Breadcrumb.vue'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Textarea from '@/components/ui/Textarea.vue'
import Modal from '@/components/ui/Modal.vue'
import FormSection from '@/components/ui/FormSection.vue'
import Rating from '@/components/ui/Rating.vue'
import Slider from '@/components/ui/Slider.vue'

const showEditModal = ref(false)
const editingConstellation = ref(null)

const constellations = ref([
  { id: 1, name: '白羊座', symbol: '♈', dateRange: '3.21-4.19', overallLuck: 4, loveLuck: 3, careerLuck: 4 },
  { id: 2, name: '金牛座', symbol: '♉', dateRange: '4.20-5.20', overallLuck: 3, loveLuck: 4, careerLuck: 3 },
  { id: 3, name: '双子座', symbol: '♊', dateRange: '5.21-6.21', overallLuck: 4, loveLuck: 4, careerLuck: 3 },
  { id: 4, name: '巨蟹座', symbol: '♋', dateRange: '6.22-7.22', overallLuck: 3, loveLuck: 3, careerLuck: 4 },
  { id: 5, name: '狮子座', symbol: '♌', dateRange: '7.23-8.22', overallLuck: 5, loveLuck: 4, careerLuck: 5 },
  { id: 6, name: '处女座', symbol: '♍', dateRange: '8.23-9.22', overallLuck: 3, loveLuck: 3, careerLuck: 4 },
  { id: 7, name: '天秤座', symbol: '♎', dateRange: '9.23-10.23', overallLuck: 4, loveLuck: 5, careerLuck: 3 },
  { id: 8, name: '天蝎座', symbol: '♏', dateRange: '10.24-11.22', overallLuck: 4, loveLuck: 4, careerLuck: 4 },
  { id: 9, name: '射手座', symbol: '♐', dateRange: '11.23-12.21', overallLuck: 5, loveLuck: 3, careerLuck: 4 },
  { id: 10, name: '摩羯座', symbol: '♑', dateRange: '12.22-1.19', overallLuck: 3, loveLuck: 3, careerLuck: 5 },
  { id: 11, name: '水瓶座', symbol: '♒', dateRange: '1.20-2.18', overallLuck: 4, loveLuck: 4, careerLuck: 3 },
  { id: 12, name: '双鱼座', symbol: '♓', dateRange: '2.19-3.20', overallLuck: 4, loveLuck: 5, careerLuck: 3 }
])

const editForm = reactive({ overallLuck: 3, loveLuck: 3, careerLuck: 3, wealthLuck: 3, dailyHoroscope: '', luckyItem: '' })

const editConstellation = (item) => {
  editingConstellation.value = item
  Object.assign(editForm, { overallLuck: item.overallLuck, loveLuck: item.loveLuck, careerLuck: item.careerLuck, wealthLuck: 3, dailyHoroscope: '', luckyItem: '' })
  showEditModal.value = true
}

const handleSave = () => { showEditModal.value = false }
</script>
