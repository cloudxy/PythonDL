<template>
  <div class="card p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
      <slot name="actions" />
    </div>
    <div class="space-y-3">
      <div
        v-for="(item, index) in items"
        :key="index"
        class="flex items-center justify-between p-3 rounded-lg hover:bg-secondary-50 transition-colors"
      >
        <div class="flex items-center gap-3">
          <div
            :class="[
              'w-2 h-2 rounded-full',
              item.active ? 'bg-success-500' : 'bg-secondary-300'
            ]"
          ></div>
          <span class="text-sm text-secondary-700">{{ item.label }}</span>
        </div>
        <span class="text-sm font-medium text-secondary-900">{{ item.value }}</span>
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
  }
})
</script>
