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
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm text-secondary-700">{{ item.label }}</span>
          <span class="text-sm font-medium text-secondary-900">{{ item.value }}</span>
        </div>
        <div class="h-2 bg-secondary-200 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all duration-300"
            :class="colorClass(item.color)"
            :style="{ width: `${item.percentage}%` }"
          ></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
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

const colorClass = (color = 'primary') => {
  const classes = {
    primary: 'bg-primary-500',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
    secondary: 'bg-secondary-500'
  }
  return classes[color] || 'bg-primary-500'
}
</script>
