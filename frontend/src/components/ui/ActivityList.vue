<template>
  <div class="card">
    <div class="p-4 border-b border-secondary-100 flex items-center justify-between">
      <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
      <slot name="header-actions" />
    </div>
    <div class="divide-y divide-secondary-100">
      <div
        v-for="(item, index) in items"
        :key="index"
        class="p-4 hover:bg-secondary-50 cursor-pointer transition-colors"
        @click="$emit('select', item)"
      >
        <div class="flex items-start gap-3">
          <div
            :class="[
              'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
              item.iconBg || 'bg-secondary-100'
            ]"
          >
            <component v-if="item.icon" :is="item.icon" class="w-5 h-5" :class="item.iconClass || 'text-secondary-600'" />
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex items-start justify-between">
              <div>
                <p class="text-sm font-medium text-secondary-900">{{ item.title }}</p>
                <p v-if="item.description" class="text-sm text-secondary-500 mt-0.5">{{ item.description }}</p>
              </div>
              <span class="text-xs text-secondary-400 flex-shrink-0 ml-2">{{ item.time }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div v-if="showMore" class="p-3 text-center border-t border-secondary-100">
      <button class="text-sm text-primary-600 hover:text-primary-700 font-medium">
        查看全部
      </button>
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
  showMore: {
    type: Boolean,
    default: false
  }
})

defineEmits(['select'])
</script>
