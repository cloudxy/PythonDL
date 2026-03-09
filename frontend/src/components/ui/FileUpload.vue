<template>
  <div class="w-full">
    <label v-if="label" class="form-label">
      {{ label }}
    </label>
    <div class="relative">
      <input
        type="file"
        :accept="accept"
        :multiple="multiple"
        :disabled="disabled"
        class="hidden"
        ref="fileInput"
        @change="handleChange"
      />
      <div
        :class="[
          'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
          dragging ? 'border-primary-500 bg-primary-50' : 'border-secondary-300 hover:border-primary-400',
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        ]"
        @click="handleClick"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="handleDrop"
      >
        <svg class="w-12 h-12 mx-auto text-secondary-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <p class="text-sm text-secondary-600">
          <span class="text-primary-600 font-medium">点击上传</span> 或拖拽文件到此处
        </p>
        <p class="text-xs text-secondary-500 mt-1">{{ hint }}</p>
      </div>
    </div>
    <!-- File list -->
    <div v-if="files.length > 0" class="mt-3 space-y-2">
      <div
        v-for="(file, index) in files"
        :key="index"
        class="flex items-center justify-between p-3 bg-secondary-50 rounded-lg"
      >
        <div class="flex items-center gap-3">
          <svg class="w-5 h-5 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span class="text-sm text-secondary-700">{{ file.name }}</span>
          <span class="text-xs text-secondary-500">{{ formatSize(file.size) }}</span>
        </div>
        <button
          @click="removeFile(index)"
          class="p-1 text-secondary-400 hover:text-danger-500 transition-colors"
        >
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  modelValue: {
    type: Array,
    default: () => []
  },
  label: {
    type: String,
    default: ''
  },
  accept: {
    type: String,
    default: ''
  },
  multiple: {
    type: Boolean,
    default: false
  },
  disabled: {
    type: Boolean,
    default: false
  },
  hint: {
    type: String,
    default: '支持常见文件格式'
  }
})

const emit = defineEmits(['update:modelValue'])

const fileInput = ref(null)
const dragging = ref(false)
const files = ref([...props.modelValue])

const handleClick = () => {
  if (!props.disabled) {
    fileInput.value?.click()
  }
}

const handleChange = (event) => {
  const newFiles = Array.from(event.target.files)
  addFiles(newFiles)
}

const handleDrop = (event) => {
  dragging.value = false
  const newFiles = Array.from(event.dataTransfer.files)
  addFiles(newFiles)
}

const addFiles = (newFiles) => {
  if (props.multiple) {
    files.value = [...files.value, ...newFiles]
  } else {
    files.value = newFiles.slice(0, 1)
  }
  emit('update:modelValue', files.value)
}

const removeFile = (index) => {
  files.value.splice(index, 1)
  emit('update:modelValue', files.value)
}

const formatSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script>
