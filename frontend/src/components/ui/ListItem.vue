<template>
  <div class="space-y-3">
    <div
      v-for="(item, index) in items"
      :key="index"
      class="flex items-center gap-3 p-3 rounded-lg hover:bg-secondary-50 cursor-pointer transition-colors"
      :class="{ 'bg-secondary-50': activeIndex === index }"
      @click="$emit('select', index)"
    >
      <div
        :class="[
          'w-10 h-10 rounded-lg flex items-center justify-center',
          item.iconBg || 'bg-secondary-100'
        ]"
      >
        <component v-if="item.icon" :is="item.icon" class="w-5 h-5" :class="item.iconClass || 'text-secondary-600'" />
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-sm font-medium text-secondary-900 truncate">{{ item.title }}</p>
        <p v-if="item.subtitle" class="text-xs text-secondary-500 truncate">{{ item.subtitle }}</p>
      </div>
      <Badge v-if="item.badge" :variant="item.badgeVariant || 'primary'" size="sm">
        {{ item.badge }}
      </Badge>
    </div>
  </div>
</template>

<script setup>
import Badge from './Badge.vue'

defineProps({
  items: {
    type: Array,
    default: () => []
  },
  activeIndex: {
    type: Number,
    default: -1
  }
})

defineEmits(['select'])
</script>
