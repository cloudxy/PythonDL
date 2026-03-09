<template>
  <div class="card p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
      <Dropdown
        v-if="actions.length"
        :items="actions"
        label="操作"
        size="sm"
        @select="$emit('action', $event)"
      />
    </div>
    <div class="space-y-4">
      <div v-for="(item, index) in items" :key="index">
        <div class="flex items-center justify-between mb-1">
          <span class="text-sm text-secondary-700">{{ item.label }}</span>
          <span class="text-sm font-medium text-secondary-900">{{ item.value }}%</span>
        </div>
        <Progress :value="item.value" :color="item.color || 'primary'" />
      </div>
    </div>
  </div>
</template>

<script setup>
import Progress from './Progress.vue'
import Dropdown from './Dropdown.vue'

defineProps({
  title: {
    type: String,
    required: true
  },
  items: {
    type: Array,
    default: () => []
  },
  actions: {
    type: Array,
    default: () => []
  }
})

defineEmits(['action'])
</script>
