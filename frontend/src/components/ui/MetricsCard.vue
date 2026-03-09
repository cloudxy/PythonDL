<template>
  <div class="card p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
      <Dropdown
        v-if="timeOptions.length"
        :items="timeOptions"
        :label="selectedTime"
        size="sm"
        position="bottom-end"
        @select="handleTimeSelect"
      />
    </div>
    <div class="space-y-4">
      <div v-for="(item, index) in metrics" :key="index">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-secondary-600">{{ item.label }}</span>
          <span class="text-sm font-semibold text-secondary-900">{{ formatValue(item.value) }}</span>
        </div>
        <div class="flex items-center gap-2">
          <Progress :value="item.percentage" :color="item.color || 'primary'" />
          <span class="text-xs text-secondary-500 w-12 text-right">{{ item.percentage }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Progress from './Progress.vue'
import Dropdown from './Dropdown.vue'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  metrics: {
    type: Array,
    default: () => []
  },
  timeOptions: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['time-change'])

const selectedTime = ref('本周')

const handleTimeSelect = (item) => {
  selectedTime.value = item.label
  emit('time-change', item.value)
}

const formatValue = (value) => {
  if (typeof value === 'number') {
    return value.toLocaleString()
  }
  return value
}
</script>
